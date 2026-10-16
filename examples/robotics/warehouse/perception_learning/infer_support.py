"""Infer qualified visual-support hypotheses from measured scene geometry only.

This module reads no physics labels and no mission/goal bindings. Metric box sizes
are explicit catalog priors. A positive result is visual consistency with support,
not a contact-force measurement or proof of successful manipulation.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def infer_frame(frame):
    entities = frame["scene_graph"]["entities"]
    now = frame["timestamp"]
    parcels = [e for e in entities if e["kind"] == "parcel"]
    stations = [e for e in entities if e["kind"] == "station"]
    comparisons = []
    for parcel in parcels:
        for station in stations:
            if parcel["parent_id"] != station["parent_id"]:
                continue
            record = {
                "subject": parcel["id"],
                "object": station["id"],
                "timestamp": now,
                "status": "ambiguous",
                "predicate": "visual_support_candidate",
                "physical_contact_asserted": False,
                "evidence_sources": [parcel["source"], station["source"]],
            }
            fresh = all(
                0 <= now - e["observed_at"] <= 0.5 and e["confidence"] >= 0.7
                for e in [parcel, station]
            )
            if not fresh:
                record["reason"] = (
                    "An entity is stale, ambiguous, or insufficiently localized."
                )
                comparisons.append(record)
                continue
            p, s = np.array(parcel["position"]), np.array(station["position"])
            uncertainty = 3 * (parcel["uncertainty_m"] + station["uncertainty_m"])
            xy_margin = 0.1 - 0.025 - float(np.max(np.abs(p[:2] - s[:2])))
            height_gap = float(p[2] - 0.025 - s[2])
            record.update(
                horizontal_containment_margin_m=xy_margin,
                bottom_to_plane_gap_m=height_gap,
                combined_three_sigma_envelope_m=uncertainty,
                confidence_semantics="heuristic geometric uncertainty, not calibrated probability",
            )
            if xy_margin > uncertainty and abs(height_gap) <= 0.006 + uncertainty:
                record.update(
                    status="candidate",
                    reason="Measured parcel footprint lies within the receiver surface and its bottom is consistent with the observed plane.",
                    confidence_score=min(parcel["confidence"], station["confidence"]),
                )
            elif xy_margin < -uncertainty or abs(height_gap) > 0.006 + uncertainty:
                record.update(
                    status="inconsistent",
                    reason="Measured geometry is separated beyond the stated uncertainty envelope.",
                )
            comparisons.append(record)
    return {
        "frame_index": frame["frame_index"],
        "timestamp": now,
        "comparisons": comparisons,
    }


def infer(composition):
    return {
        "schema": "tensorcode.warehouse.visual-support-candidates.v1",
        "scope": "Geometry-only support hypotheses; not tactile/contact evidence; not inferred from requested task destinations",
        "priors": {
            "parcel_half_size_m": 0.025,
            "receiver_half_extent_m": 0.1,
            "plane_tolerance_m": 0.006,
        },
        "frames": [infer_frame(frame) for frame in composition["frames"]],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("composition", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = infer(json.loads(args.composition.read_text()))
    result["source_composition_sha256"] = hashlib.sha256(
        args.composition.read_bytes()
    ).hexdigest()
    result["implementation_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    args.output.write_text(json.dumps(result, indent=2) + "\n")
