import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples/robotics"))
from warehouse.perception_learning.infer_support import infer_frame


def frame(position=(0.0, 0.0, 0.925), confidence=0.9):
    def entity(name, kind, p, score):
        return {
            "id": name,
            "kind": kind,
            "position": p,
            "parent_id": "cell",
            "observed_at": 1.0,
            "confidence": score,
            "uncertainty_m": 0.002,
            "source": "sensor-frame#" + name,
        }

    return {
        "frame_index": 1,
        "timestamp": 1.0,
        "scene_graph": {
            "entities": [
                entity("p", "parcel", position, confidence),
                entity("s", "station", (0.0, 0.0, 0.9), 0.9),
            ]
        },
    }


def test_near_plane_yields_hypothesis_not_contact_assertion():
    result = infer_frame(frame())["comparisons"][0]
    assert result["status"] == "candidate"
    assert result["predicate"] == "visual_support_candidate"
    assert result["physical_contact_asserted"] is False


def test_separated_geometry_and_weak_evidence_do_not_gain_support():
    assert (
        infer_frame(frame((0.0, 0.3, 0.925)))["comparisons"][0]["status"]
        == "inconsistent"
    )
    assert (
        infer_frame(frame((0.0, 0.0, 1.2)))["comparisons"][0]["status"]
        == "inconsistent"
    )
    assert infer_frame(frame(confidence=0.4))["comparisons"][0]["status"] == "ambiguous"


def test_requested_destinations_do_not_change_geometric_inference():
    observation = frame()
    before = infer_frame(observation)
    observation["mission_grounding"] = {
        "tasks": [{"object_id": "p", "destination_id": "different_station"}]
    }
    assert infer_frame(observation) == before
