"""Ground compositional queries in measured multi-color scene observations.

Calibration defines workcell regions. Color/shape observations define children.
No evaluation poses, task outcome labels or simulator object IDs are read.
"""

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from warehouse.language import (
    GroundingError,
    Instruction,
    answer_question,
    ground_instruction,
    parse_instruction,
)
from warehouse.perception_learning.scene_observer import observe_views
from warehouse.representation import Entity, SceneGraph

COLOR_JOB = {"orange": "a", "magenta": "a", "green": "b", "yellow": "b", "blue": "b"}


def compose(dataset, receiver_b_color="blue", mission_request=None):
    manifest = json.loads((dataset / "manifest.json").read_text())
    frame_indices = sorted({row["frame_index"] for row in manifest["frames"]})
    result = {
        "source_run": manifest["run"],
        "source": "fresh_rgbd_plus_declared_color_shape_and_workcell_calibration_priors",
        "identity_prior": "orange/magenta ->job a;green/yellow-or-blue ->job b, only when unique in observed cell",
        "frames": [],
    }
    for frame_index in frame_indices:
        scene = SceneGraph()
        lanes = {}
        for row in manifest["frames"]:
            if row["frame_index"] == frame_index:
                lanes.setdefault(row["lane"], []).append(row)
        observations = {}
        now = None
        for lane, entries in sorted(lanes.items()):
            frames = []
            for entry in sorted(entries, key=lambda x: x["view"]):
                path = dataset / entry["path"]
                contract = json.loads((path / "sensor-contract.json").read_text())
                frames.append(
                    {
                        "rgb": np.array(Image.open(path / "sensor-rgb.png")),
                        "depth": np.load(path / "sensor-depth.npz")["depth"],
                        "calibration": contract["camera"],
                        "timestamp": entry["timestamp"],
                    }
                )
            now = entries[0]["timestamp"]
            center = np.array(frames[0]["calibration"]["position"]) + 2.5 * np.array(
                frames[0]["calibration"]["forward"]
            )
            evidence = observe_views(
                frames,
                now,
                workcell_center_xy=center[:2],
                workcell_halfwidth_m=contract.get("workcell_halfwidth_m", 0.7),
                colors=tuple(COLOR_JOB),
            )
            observations[str(lane)] = evidence
            cell = f"workcell_{lane}"
            source = f"composition.json#frame-{frame_index}/lane-{lane}"
            scene.upsert(
                Entity(
                    cell,
                    "workcell",
                    {"basis": "calibrated_region", "lane": str(lane)},
                    tuple(center),
                    now,
                    source + "/calibration",
                    1.0,
                    0.0,
                )
            )
            groups = {}
            for track in evidence["tracks"]:
                groups.setdefault((track["kind"], track["color"]), []).append(track)
            for (kind, color), tracks in groups.items():
                for index, track in enumerate(tracks):
                    unique = len(tracks) == 1
                    expected_role = (
                        kind == "parcel" and color in {"orange", "green"}
                    ) or (kind == "receiver" and color in {"magenta", receiver_b_color})
                    if unique and expected_role:
                        identifier = f"{kind}_{lane}_{COLOR_JOB[color]}"
                    elif unique:
                        identifier = f"{kind}_{color}_{lane}"
                    else:
                        identifier = f"{kind}_{color}_{lane}_candidate{index}"
                    confidence = (
                        track["confidence_score"]
                        if unique
                        else min(track["confidence_score"], 0.49)
                    )
                    position = np.array(track["center_xyz"])
                    if kind == "receiver":
                        position[2] += track["dimensions_xyz"][2] / 2
                    scene.upsert(
                        Entity(
                            identifier,
                            "station" if kind == "receiver" else kind,
                            {
                                "color": color,
                                "lane": str(lane),
                                "identity": "unique_color_shape_in_calibrated_cell"
                                if unique
                                else "ambiguous",
                            },
                            tuple(position),
                            now,
                            source + "/" + track["id"],
                            confidence,
                            max(track["position_std_xyz_m"]),
                            parent_id=cell,
                        )
                    )
        questions = ["how many parcels?"]
        for lane in sorted(lanes):
            questions += [
                f"what is in workcell_{lane}?",
                f"where is the orange parcel in workcell_{lane}?",
                f"where is the green parcel in workcell_{lane}?",
                f"what is on the yellow station in workcell_{lane}?",
                f"what is on the blue station in workcell_{lane}?",
            ]
        answers = [
            {"question": q, "answer": asdict(answer_question(q, scene, now))}
            for q in questions
        ]
        groundings = []
        cell_trees = []
        for lane in sorted(lanes):
            instruction = f"move the orange parcel in workcell_{lane} to the magenta station in workcell_{lane} then move the green parcel in workcell_{lane} to the {receiver_b_color} station in workcell_{lane}"
            tree = parse_instruction(instruction)
            cell_trees.append(tree)
            entry = {
                "instruction": instruction,
                "parse_tree": tree.to_dict(),
                "execution": "not_executed; observation-grounding audit only",
            }
            try:
                entry.update(
                    status="grounded",
                    tasks=[asdict(t) for t in ground_instruction(tree, scene, now)],
                )
            except GroundingError as exc:
                entry.update(
                    status=exc.status, reason=str(exc), candidates=list(exc.candidates)
                )
            groundings.append(entry)
        mission_tree = (
            Instruction("parallel", children=tuple(cell_trees))
            if len(cell_trees) > 1
            else cell_trees[0]
        )
        if mission_request is not None:
            mission_tree = parse_instruction(mission_request["instruction"])
            if (
                json.loads(json.dumps(mission_tree.to_dict()))
                != mission_request["parse_tree"]
            ):
                raise ValueError(
                    "Mission instruction does not match declared parse tree"
                )
        mission = {
            "composition": "authored parallel root over independently parsed per-cell instructions",
            "parse_tree": mission_tree.to_dict(),
            "execution": "not_executed; observation-grounding audit only",
        }
        if mission_request is not None:
            mission.update(
                composition="re-grounded exact episode instruction; observation audit only",
                instruction=mission_request["instruction"],
                episode_probe_sha256=mission_request["episode_probe_sha256"],
            )
        try:
            mission.update(
                status="grounded",
                tasks=[
                    asdict(task)
                    for task in ground_instruction(mission_tree, scene, now)
                ],
            )
        except GroundingError as exc:
            mission.update(
                status=exc.status, reason=str(exc), candidates=list(exc.candidates)
            )
        result["frames"].append(
            {
                "frame_index": frame_index,
                "timestamp": now,
                "scene_graph": scene.to_dict(),
                "observations": observations,
                "questions": answers,
                "instruction_grounding": groundings,
                "mission_grounding": mission,
            }
        )
    paths = [
        Path(__file__),
        *sorted(Path(__file__).parent.glob("*.py")),
        Path(__file__).parents[1] / "perception.py",
        Path(__file__).parents[1] / "perception_tracking.py",
        Path(__file__).parents[1] / "language.py",
        Path(__file__).parents[1] / "representation.py",
    ]
    result["sources_sha256"] = {
        str(p.relative_to(Path(__file__).parents[2])): hashlib.sha256(
            p.read_bytes()
        ).hexdigest()
        for p in paths
    }
    result["sensor_files_sha256"] = {
        str(p.relative_to(dataset)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(dataset.glob("frame-*/*"))
        if p.is_file()
    }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--receiver-b-color", choices=["yellow", "blue"], default="blue"
    )
    parser.add_argument(
        "--mission-request",
        type=Path,
        help="Instruction-only JSON: instruction, parse_tree, episode_probe_sha256; no outcome/pose inputs",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            compose(
                args.dataset,
                args.receiver_b_color,
                json.loads(args.mission_request.read_text())
                if args.mission_request
                else None,
            ),
            indent=2,
        )
        + "\n"
    )
