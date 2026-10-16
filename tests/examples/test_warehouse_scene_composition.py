"""Language grounding must preserve sensor ambiguity and declared color roles."""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples/robotics"))
from warehouse.perception_learning import compose_capture as module


def dataset(tmp_path):
    frames = []
    for view in range(3):
        name = f"frame-00000-r0-v{view}"
        path = tmp_path / name
        path.mkdir()
        Image.fromarray(np.zeros((2, 2, 3), np.uint8)).save(path / "sensor-rgb.png")
        np.savez(path / "sensor-depth.npz", depth=np.ones((2, 2)))
        calibration = {
            "position": [1.1, -0.25, 3.15],
            "forward": [0, 0, -1],
            "right": [1, 0, 0],
            "up": [0, 1, 0],
            "focal_pixels": 2,
            "width": 2,
            "height": 2,
        }
        (path / "sensor-contract.json").write_text(
            json.dumps({"camera": calibration, "workcell_halfwidth_m": 0.9})
        )
        frames.append(
            {"path": name, "frame_index": 0, "lane": 0, "view": view, "timestamp": 0.0}
        )
    (tmp_path / "manifest.json").write_text(
        json.dumps({"run": "sensor-fixture", "frames": frames})
    )
    return tmp_path


def track(kind, color, index):
    return {
        "id": f"track{index}",
        "kind": kind,
        "color": color,
        "center_xyz": [1.2, -0.2 + index * 0.1, 0.8],
        "dimensions_xyz": [0.05, 0.05, 0.05] if kind == "parcel" else [0.2, 0.2, 0.004],
        "confidence_score": 0.9,
        "position_std_xyz_m": [0.002] * 3,
    }


def test_pixel_only_scene_composes_two_task_instruction_and_abstains_contact(
    tmp_path, monkeypatch
):
    observed = [
        track("parcel", "orange", 0),
        track("parcel", "green", 1),
        track("receiver", "magenta", 2),
        track("receiver", "blue", 3),
    ]
    monkeypatch.setattr(
        module, "observe_views", lambda *args, **kwargs: {"tracks": observed}
    )
    result = module.compose(dataset(tmp_path), "blue")["frames"][0]
    grounding = result["instruction_grounding"][0]
    assert grounding["status"] == "grounded"
    assert grounding["tasks"][1]["dependencies"] == ("task-1",)
    assert grounding["tasks"][1]["object_id"] == "parcel_0_b"
    questions = {q["question"]: q["answer"]["status"] for q in result["questions"]}
    assert questions["where is the green parcel in workcell_0?"] == "supported"
    assert questions["what is on the blue station in workcell_0?"] == "unknown"
    assert not (tmp_path / "evaluation_truth.json").exists()


def test_duplicate_color_identity_is_not_silently_aliased(tmp_path, monkeypatch):
    observed = [
        track("parcel", "orange", 0),
        track("parcel", "orange", 1),
        track("parcel", "green", 2),
        track("receiver", "magenta", 3),
        track("receiver", "blue", 4),
    ]
    monkeypatch.setattr(
        module, "observe_views", lambda *args, **kwargs: {"tracks": observed}
    )
    frame = module.compose(dataset(tmp_path), "blue")["frames"][0]
    entities = frame["scene_graph"]["entities"]
    assert "parcel_0_a" not in {e["id"] for e in entities}
    assert frame["instruction_grounding"][0]["status"] == "unknown"


def test_legacy_yellow_does_not_overwrite_declared_blue_role(tmp_path, monkeypatch):
    observed = [track("receiver", "yellow", 0), track("receiver", "blue", 1)]
    monkeypatch.setattr(
        module, "observe_views", lambda *args, **kwargs: {"tracks": observed}
    )
    entities = module.compose(dataset(tmp_path), "blue")["frames"][0]["scene_graph"][
        "entities"
    ]
    by_id = {e["id"]: e for e in entities}
    assert by_id["receiver_0_b"]["attributes"]["color"] == "blue"
    assert by_id["receiver_yellow_0"]["attributes"]["color"] == "yellow"
