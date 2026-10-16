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
    data = json.loads((SOURCE / "experiments.json").read_text())
    data["experiments"][0]["episodes"][0]["inspection_sources"][0]["link"] = (
        "../../../../README.md"
    )
    with pytest.raises(ValueError, match="nonlocal"):
        builder.validate(data, builder.RESULTS)


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
    experiment["episodes"][0].pop("presentation_still", None)
    experiment["episodes"][0]["inspection_sources"] = [
        {"title": "Scene", "path": "scene.json"}
    ]
    (root / "scene.json").write_text(json.dumps({"name": experiment["summary"]}))
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
    episode = {"traces": traces, "multi_stage_tracks": True}
    builder.enrich_episode(episode, tmp_path)
    assert [track["label"] for track in episode["chapter_tracks"]] == ["Left", "Right"]
    assert episode["chapter_tracks"][0]["chapters"][-1]["label"] == "lift"
    assert episode["chapter_tracks"][1]["chapters"][0]["label"] == "failure"


def test_episode_inspection_preserves_source_and_rejects_escape(tmp_path):
    scene = {
        "warehouse": {"robots": [{"id": "g1-a"}, {"id": "g1-b"}]},
        "instruction": "<img src=x onerror=alert(1)>",
    }
    (tmp_path / "scene.json").write_text(json.dumps(scene))
    episode = {
        "inspection_sources": [{"title": "Scene hierarchy", "path": "scene.json"}]
    }
    builder.enrich_episode(episode, tmp_path)
    assert episode["inspections"][0]["data"] == scene
    assert episode["inspections"][0]["path"] == "scene.json"
    episode["inspection_sources"][0]["fields"] = ["instruction"]
    builder.enrich_episode(episode, tmp_path)
    assert episode["inspections"][0]["data"] == {"instruction": scene["instruction"]}

    episode["inspection_sources"][0]["path"] = "../outside.json"
    with pytest.raises(ValueError, match="nonlocal"):
        builder.enrich_episode(episode, tmp_path)


def test_nested_robot_trace_and_independently_counted_pair(tmp_path):
    records = [
        {
            "t": 21.84,
            "robots": {
                "r0": {
                    "phase": "reacquire_retract",
                    "controller_state": {"recovery_attempts": 1},
                    "sensor_state": {"pad_force_n": [2.0266, 0]},
                },
                "r1": {"phase": "complete"},
            },
        }
    ]
    (tmp_path / "records.json").write_text(json.dumps(records))
    for name, successes in [("recovery", [True, True]), ("baseline", [False, True])]:
        (tmp_path / f"{name}.json").write_text(
            json.dumps(
                {
                    "robots": {
                        str(i): {"strict_current_settle_success": success}
                        for i, success in enumerate(successes)
                    }
                }
            )
        )
    episode = {
        "traces": [{"path": "records.json", "robot": "r0", "label": "Recovery right"}],
        "metrics_source": {
            "path": "recovery.json",
            "fields": [
                {
                    "key": "robots",
                    "label": "strict settled",
                    "count_where": {"strict_current_settle_success": True},
                    "compare_path": "baseline.json",
                    "compare_label": "baseline",
                }
            ],
        },
    }
    builder.enrich_episode(episode, tmp_path)
    assert episode["chapters"][0]["label"] == "reacquire retract"
    assert episode["chapters"][0]["time"] == 21.84
    assert episode["metrics"][0]["value"] == 2
    assert episode["metrics"][0]["reference"] == 1
    assert any(
        s["name"].endswith("recovery_attempts") and s["value"] == 1
        for s in episode["signals"]
    )
    assert not any(s["value"] == "complete" for s in episode["signals"])


def test_repeated_jobs_keep_distinct_chapters_and_nested_success_counts(tmp_path):
    (tmp_path / "records.json").write_text(
        json.dumps(
            [
                {"t": 1, "robots": {"r0": {"phase": "walk", "current_job": "a"}}},
                {"t": 70, "robots": {"r0": {"phase": "walk", "current_job": "b"}}},
            ]
        )
    )
    episode = {
        "traces": [
            {"path": "records.json", "robot": "r0", "label": "r0", "job_labels": True}
        ]
    }
    builder.enrich_episode(episode, tmp_path)
    assert [(c["time"], c["label"]) for c in episode["chapters"]] == [
        (1, "A · walk"),
        (70, "B · walk"),
    ]
    validation = {
        "r0": {"a": {"physical_success": True}, "b": {"physical_success": False}},
        "r1": {"a": {"physical_success": True}},
    }
    assert (
        builder.metric_value(
            validation, {"count_depth": 2, "count_where": {"physical_success": True}}
        )
        == 2
    )
    with pytest.raises(KeyError):
        builder.metric_value(
            validation, {"count_depth": 2, "count_where": {"invented": True}}
        )


def test_presentation_excerpt_uses_source_offset_and_excludes_future(tmp_path):
    (tmp_path / "records.json").write_text(
        json.dumps(
            [
                {"t": 19, "stage": "approach"},
                {"t": 20, "stage": "close"},
                {"t": 21.84, "stage": "reacquire"},
                {"t": 40, "stage": "future"},
                {"t": 85, "stage": "complete"},
            ]
        )
    )
    episode = {
        "traces": [
            {
                "path": "records.json",
                "label": "Source +20 s",
                "video_start_s": 20,
                "video_end_s": 40,
            }
        ]
    }
    builder.enrich_episode(episode, tmp_path)
    assert [c["label"] for c in episode["chapters"]] == ["close", "reacquire"]
    assert episode["chapters"][1]["time"] == pytest.approx(1.84)
    assert all(0 <= sample["time"] < 20 for sample in episode["signals"])


def test_policy_audit_distinguishes_missing_invalid_and_physical_failures(tmp_path):
    audited = {
        "policy": "authored",
        "case": "c0",
        "status": "audited",
        "episode_success": False,
        "recorded_seconds": 23.4,
        "termination_reason": "fall",
        "robots": {"r0": {"strict_success": False}},
    }
    report = {
        "expected_runs": 4,
        "complete_and_matched": False,
        "matching_errors": [],
        "scope": "fixture only",
        "runs": [
            audited,
            *[
                {"policy": status, "case": "c0", "status": status}
                for status in ["invalid", "incomplete", "unrun"]
            ],
        ],
    }
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(report))
    result = builder.load_policy_comparison(tmp_path, "audit.json")
    assert [row["status"] for row in result["rows"]] == [
        "physical failure",
        "invalid",
        "incomplete",
        "unrun",
    ]
    assert result["rows"][0]["seconds"] == 23.4
    assert "seconds" not in result["rows"][1]
    report["complete_and_matched"] = True
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="incomplete or unmatched"):
        builder.load_policy_comparison(tmp_path, "audit.json")
    report.update(expected_runs=1, runs=[audited])
    path.write_text(json.dumps(report))
    assert builder.load_policy_comparison(tmp_path, "audit.json")[
        "complete_and_matched"
    ]
    report["matching_errors"] = ["initial state differs"]
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="unmatched"):
        builder.load_policy_comparison(tmp_path, "audit.json")


def test_piecewise_presentation_retains_cuts_and_source_clocks(tmp_path):
    timeline = {
        "sequence": {
            "chapters": [
                {
                    "label": "first",
                    "video_start_s": 0,
                    "video_end_s": 8,
                    "source_start_s": 0,
                    "source_end_s": 8,
                },
                {
                    "label": "second",
                    "video_start_s": 8,
                    "video_end_s": 16,
                    "source_start_s": 35,
                    "source_end_s": 43,
                },
            ]
        }
    }
    (tmp_path / "movie.json").write_text(json.dumps(timeline))
    (tmp_path / "records.json").write_text(
        json.dumps(
            [
                {"t": 0, "stage": "walk"},
                {"t": 7, "stage": "approach"},
                {"t": 20, "stage": "omitted"},
                {"t": 35, "stage": "approach"},
                {"t": 42, "stage": "lift"},
                {"t": 80, "stage": "future"},
            ]
        )
    )
    episode = {
        "timeline_source": "movie.json",
        "traces": [{"path": "records.json", "label": "r0"}],
    }
    builder.enrich_episode(episode, tmp_path)
    assert [(c["time"], c["label"]) for c in episode["chapters"]] == [
        (0, "Cut 1 · walk"),
        (7, "approach"),
        (8, "Cut 2 · approach"),
        (15, "lift"),
    ]
    assert any(
        s.get("source_time") == 35 and s["time"] == 8 for s in episode["signals"]
    )
    assert not any(s.get("source_time") == 20 for s in episode["signals"])
    timeline["sequence"]["chapters"][1]["video_start_s"] = 9
    (tmp_path / "movie.json").write_text(json.dumps(timeline))
    with pytest.raises(ValueError, match="contiguous"):
        builder.presentation_timeline(tmp_path, "movie.json")


def test_native_comparison_expands_all_clips_and_blocks_unfinished_arms(tmp_path):
    (tmp_path / "native").mkdir()
    (tmp_path / "native/clip.mp4").write_bytes(b"fixture")
    report = {
        "counts": [
            {"arm": "psi01", "condition": "nominal", "status": "complete"},
            {"arm": "bc", "condition": "nominal", "status": "unrun"},
        ],
        "episodes": [
            {
                "id": "psi01-nominal-c0",
                "arm": "psi01",
                "condition": "nominal",
                "config": 0,
                "success": False,
                "steps": 10,
                "simulated_seconds": 0.2,
                "video": "clip.mp4",
            }
        ],
        "paired_outcomes": [{"condition": "nominal", "config": 0, "arms": {}}],
        "training_comparison": "configuration-matched only",
        "verification_scope": {"initial_state": "partial reset only"},
        "limitations": "fixture",
    }
    (tmp_path / "native/results.json").write_text(json.dumps(report))
    experiment = {
        "native_comparison_source": "native/results.json",
        "native_required_arms": ["psi01", "bc"],
    }
    with pytest.raises(ValueError, match="not complete"):
        builder.expand_native_comparison(experiment, tmp_path)
    report["episodes"].append(
        {**report["episodes"][0], "id": "bc-nominal-c0", "arm": "bc", "video": None}
    )
    (tmp_path / "native/results.json").write_text(json.dumps(report))
    experiment["native_allow_partial"] = True
    builder.expand_native_comparison(experiment, tmp_path)
    assert len(experiment["episodes"]) == 1
    assert "failure" in experiment["episodes"][0]["title"]
    assert experiment["episodes"][0]["video"] == "native/clip.mp4"
    assert experiment["native_comparison"]["counts"][1]["status"] == "unrun"
    assert (
        experiment["native_comparison"]["verification_scope"]["initial_state"]
        == "partial reset only"
    )


def test_placement_comparison_preserves_regression_and_unpaired_condition(tmp_path):
    report = {
        "rows": [
            {
                "id": "xplus",
                "label": "X+20 mm",
                "v5": {"placements": 6, "of": 6},
                "v6": {"placements": 5, "of": 6},
            },
            {
                "id": "yminus",
                "label": "Y−20 mm",
                "v5": None,
                "v6": {"placements": 6, "of": 6},
            },
        ],
        "matched": {"v5_episodes": 2, "v6_episodes": 2, "episodes": 3},
        "scope": "development only",
    }
    path = tmp_path / "comparison.json"
    path.write_text(json.dumps(report))
    episode = {"placement_comparison_source": "comparison.json"}
    builder.enrich_episode(episode, tmp_path)
    assert episode["placement_comparison"]["rows"][0]["v6"]["placements"] == 5
    assert episode["placement_comparison"]["rows"][1]["v5"] is None
    report["rows"][0]["v6"]["placements"] = 7
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="Invalid placement"):
        builder.enrich_episode(episode, tmp_path)
