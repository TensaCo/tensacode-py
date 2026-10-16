"""Recorded experiments must remain portable, connected and truth-labeled."""

import importlib.util
import json
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[2] / "examples/robotics/lab"
spec = importlib.util.spec_from_file_location("robotics_lab_build", SOURCE / "build.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def test_current_lab_is_reproducible_and_connected(tmp_path):
    data = json.loads((SOURCE / "experiments.json").read_text())
    builder.validate(data, builder.RESULTS)
    before = (builder.RESULTS / "index.html").read_bytes()
    assert builder.build().read_bytes() == before
    assert b"ARCHIVE" not in before
    assert b"hand error (evaluation truth)" in before


def test_rejects_dangling_port_and_media_escape():
    data = json.loads((SOURCE / "experiments.json").read_text())
    data["experiments"][0]["graph"]["wires"].append(["cam.missing", "det.rgb"])
    with pytest.raises(ValueError, match="Unknown wire endpoint"):
        builder.validate(data, builder.RESULTS)
    with pytest.raises(ValueError, match="nonlocal"):
        builder.relative_file(builder.RESULTS, "../../../../README.md")


def test_trace_clock_and_evaluation_truth(tmp_path):
    (tmp_path / "records.json").write_text(
        json.dumps(
            [
                {"t": 3, "stage": "walk", "hand": [0, 0, 0], "desired_hand": []},
                {
                    "t": 4,
                    "stage": "reach",
                    "hand": [0, 0, 0],
                    "desired_hand": [0.01, 0, 0],
                    "pad_force_n": {"pad_a": 2},
                },
                {
                    "t": 5,
                    "stage": "close",
                    "pad_force_n": {"pad_a": 2},
                    "sensor_state": {"pad_force_n": [5, 6]},
                },
            ]
        )
    )
    episode = {
        "traces": [
            {
                "path": "records.json",
                "label": "Left policy",
                "block": "traj",
                "video_start_s": 3,
            }
        ]
    }
    builder.enrich_episode(episode, tmp_path)
    assert episode["events"][:2] == [
        {"time": 0, "label": "Left policy: walk", "block": "traj"},
        {"time": 1, "label": "Left policy: reach", "block": "traj"},
    ]
    assert episode["chapters"] == [
        {"time": 0, "label": "walk", "block": "traj"},
        {"time": 1, "label": "reach", "block": "traj"},
        {"time": 2, "label": "close", "block": "traj"},
    ]
    error = next(s for s in episode["signals"] if "evaluation truth" in s["name"])
    assert error["value"] == 10
    assert error["time"] == 1
    assert error["unit"] == "mm"
    forces = {s["name"]: s["value"] for s in episode["signals"] if "force" in s["name"]}
    assert forces["Left policy · pad_a cube-contact force (evaluation truth)"] == 2
    assert forces["Left policy · pad_a measured force"] == 5
    assert forces["Left policy · pad_b measured force"] == 6


def test_embedded_metadata_cannot_close_script(tmp_path):
    source = tmp_path / "source"
    root = tmp_path / "results"
    source.mkdir()
    root.mkdir()
    data = json.loads((SOURCE / "experiments.json").read_text())
    experiment = data["experiments"][0]
    experiment["summary"] = '</script><script>alert("unsafe")</script>'
    experiment.pop("outcomes_source", None)
    experiment.pop("outcome_collections", None)
    experiment["episodes"] = [experiment["episodes"][0]]
    experiment["episodes"][0].pop("traces", None)
    experiment["episodes"][0].pop("metrics_source", None)
    data["experiments"] = [experiment]
    for path in [
        experiment["evidence"],
        experiment["episodes"][0]["video"],
        experiment["episodes"][0]["poster"],
    ]:
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()
    for name in ("index.template.html", "lab.css", "graph.js", "app.js"):
        (source / name).write_text((SOURCE / name).read_text())
    (source / "experiments.json").write_text(json.dumps(data))
    html = builder.build(source, root).read_text()
    assert experiment["summary"] not in html
    assert "\\u003c/script>" in html


def test_episode_metrics_use_selected_recorded_run(tmp_path):
    (tmp_path / "results.json").write_text(
        json.dumps({"runs": [{"id": "a", "lift": 0.12}, {"id": "b", "lift": 0.03}]})
    )
    episode = {
        "metrics_source": {
            "path": "results.json",
            "run_id": "b",
            "fields": [
                {"key": "lift", "label": "peak lift", "scale": 100, "unit": "cm"}
            ],
        }
    }
    builder.enrich_episode(episode, tmp_path)
    assert episode["metrics"] == [{"label": "peak lift", "value": 3.0, "unit": "cm"}]
    episode["metrics_source"]["fields"][0]["compare_run_id"] = "a"
    builder.enrich_episode(episode, tmp_path)
    assert episode["metrics"][0]["reference"] == 12.0
    assert episode["metrics"][0]["value"] == 3.0
    episode["metrics_source"]["fields"][0]["key"] = "unmeasured"
    with pytest.raises(KeyError):
        builder.enrich_episode(episode, tmp_path)


def test_paired_chapters_follow_only_selected_trace(tmp_path):
    (tmp_path / "left.json").write_text(
        json.dumps([{"t": 0, "stage": "walk"}, {"t": 1, "stage": "lift"}])
    )
    (tmp_path / "right.json").write_text(json.dumps([{"t": 0, "stage": "failure"}]))
    traces = [
        {"path": "left.json", "label": "Left"},
        {"path": "right.json", "label": "Right"},
    ]
    episode = {"traces": traces}
    builder.enrich_episode(episode, tmp_path)
    assert [c["label"] for c in episode["chapters"]] == ["walk", "lift"]
    traces[1]["chapters"] = True
    episode = {"traces": traces}
    builder.enrich_episode(episode, tmp_path)
    assert [c["label"] for c in episode["chapters"]] == ["failure"]
    assert episode["chapter_label"] == "Right"
