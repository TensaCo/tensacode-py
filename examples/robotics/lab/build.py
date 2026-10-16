"""Build the standalone, file://-compatible robotics diagnosis application."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
from pathlib import Path

SOURCE = Path(__file__).resolve().parent
RESULTS = SOURCE.parents[2] / "docs/results/robotics"


def relative_file(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"Missing or nonlocal evidence/media: {value}")
    return path


def validate(data: dict, root: Path) -> None:
    if data.get("schema_version") != 1 or not data.get("experiments"):
        raise ValueError("Expected schema_version 1 and nonempty experiments")
    ids = set()
    for experiment in data["experiments"]:
        identifier = experiment["id"]
        if not re.fullmatch(r"[a-z0-9-]+", identifier) or identifier in ids:
            raise ValueError(f"Invalid or duplicate experiment ID: {identifier}")
        ids.add(identifier)
        if experiment["status"] not in {"pass", "fail", "ref"}:
            raise ValueError("Status must be pass, fail or ref")
        relative_file(root, experiment["evidence"])
        if experiment.get("outcomes_source"):
            relative_file(root, experiment["outcomes_source"])
        graph = experiment["graph"]
        for identifier, block in graph["blocks"].items():
            if not re.fullmatch(r"[a-zA-Z0-9_-]+", identifier):
                raise ValueError(f"Invalid block ID: {identifier}")
            if block["kind"] not in {"o", "a", "f", "x"}:
                raise ValueError("Unknown block kind")
            for key in ("col", "row"):
                if not isinstance(block[key], (int, float)) or not math.isfinite(
                    block[key]
                ):
                    raise ValueError("Block position must be finite")
            for side in ("ins", "outs"):
                if len(set(block[side])) != len(block[side]):
                    raise ValueError("Duplicate port name")
        for start, end in graph["wires"]:
            for endpoint, side in ((start, "outs"), (end, "ins")):
                name, port = endpoint.split(".", 1)
                if (
                    name not in graph["blocks"]
                    or port not in graph["blocks"][name][side]
                ):
                    raise ValueError(f"Unknown wire endpoint: {endpoint}")
        episode_ids = set()
        if not experiment["episodes"]:
            raise ValueError("Experiment needs a recorded episode")
        for episode in experiment["episodes"]:
            if episode["id"] in episode_ids:
                raise ValueError("Duplicate episode ID")
            episode_ids.add(episode["id"])
            relative_file(root, episode["video"])
            if episode.get("poster"):
                relative_file(root, episode["poster"])
            if episode.get("presentation_still"):
                relative_file(root, episode["presentation_still"]["path"])
                relative_file(root, episode["presentation_still"]["metadata"])
            for trace in episode.get("traces", []):
                relative_file(root, trace["path"])
                for block in [
                    trace.get("block"),
                    *trace.get("signal_blocks", {}).values(),
                ]:
                    if block is not None and block not in graph["blocks"]:
                        raise ValueError(f"Unknown telemetry block: {block}")
            if (
                sum(
                    trace.get("chapters") is True for trace in episode.get("traces", [])
                )
                > 1
            ):
                raise ValueError("Choose one trace as the chapter source")
            for inspection in episode.get("inspection_sources", []):
                relative_file(root, inspection["path"])
                for key in ("image", "link"):
                    if inspection.get(key):
                        relative_file(root, inspection[key])
            if episode.get("metrics_source"):
                relative_file(root, episode["metrics_source"]["path"])


def enrich_episode(episode: dict, root: Path) -> None:
    inspections = []
    for item in episode.get("inspection_sources", []):
        data = json.loads(relative_file(root, item["path"]).read_text())
        if item.get("fields"):
            data = {field: data[field] for field in item["fields"]}
        inspection = {"title": item["title"], "path": item["path"], "data": data}
        for key in ("image", "caption", "link"):
            if item.get(key):
                inspection[key] = item[key]
        inspections.append(inspection)
    episode["inspections"] = inspections
    signals = list(episode.get("signals", []))
    events = list(episode.get("events", []))
    chapters = list(episode.get("chapters", []))
    derive_chapters = not chapters
    explicit_chapter_source = any(
        trace.get("chapters") is True for trace in episode.get("traces", [])
    )
    chapter_tracks = []
    for trace_index, trace in enumerate(episode.get("traces", [])):
        chapter_source = derive_chapters and trace.get(
            "chapters", trace_index == 0 and not explicit_chapter_source
        )
        if chapter_source:
            episode["chapter_label"] = trace["label"]
        path = relative_file(root, trace["path"])
        records = json.loads(
            gzip.decompress(path.read_bytes())
            if path.suffix == ".gz"
            else path.read_bytes()
        )
        track_chapters = []
        chapter_tracks.append({"label": trace["label"], "chapters": track_chapters})
        if trace.get("robot"):
            records = [
                {
                    **row["robots"][trace["robot"]],
                    "t": row["t"],
                    "stage": row["robots"][trace["robot"]].get("phase"),
                }
                for row in records
            ]
        last_phase = None
        last_job = None
        last_sample = -1.0
        offset = trace.get("video_start_s", 0)
        if trace.get("video_end_s") is not None:
            records = [
                row
                for row in records
                if offset <= float(row["t"]) < trace["video_end_s"]
            ]
        for row in records:
            time = float(row["t"]) - offset
            if time < 0:
                continue
            state = row.get("controller_state") or {}
            phase = state.get("phase") or row.get("stage")
            job = row.get("current_job") if trace.get("job_labels") else None
            chapter_label = str(phase).replace("_", " ")
            if job is not None:
                chapter_label = f"{job.upper()} · {chapter_label}"
            if phase != last_phase or job != last_job:
                events.append(
                    {
                        "time": time,
                        "label": f"{trace['label']}: {phase}",
                        "block": trace.get("block"),
                    }
                )
                if phase:
                    track_chapters.append(
                        {
                            "time": time,
                            "label": chapter_label,
                            "block": trace.get("block"),
                        }
                    )
                if chapter_source and phase:
                    chapters.append(
                        {
                            "time": time,
                            "label": chapter_label,
                            "block": trace.get("block"),
                        }
                    )
                last_phase = phase
                last_job = job
            if time - last_sample < 0.2:
                continue
            last_sample = time
            values = {"phase": (phase, "")}
            desired = row.get("desired_hand", [])
            if len(desired) == 3 and len(row.get("hand", [])) == 3:
                values["hand error (evaluation truth)"] = (
                    round(math.dist(desired, row["hand"]) * 1000, 1),
                    "mm",
                )
            measured_forces = (row.get("sensor_state") or {}).get("pad_force_n")
            if measured_forces is not None:
                for pad, force in zip(("pad_a", "pad_b"), measured_forces):
                    values[pad + " measured force"] = (round(force, 2), "N")
            else:
                for pad, force in row.get("pad_force_n", {}).items():
                    values[pad + " cube-contact force (evaluation truth)"] = (
                        round(force, 2),
                        "N",
                    )
            for name in (
                "tracking_error_m",
                "orientation_error_rad",
                "clearance_m",
                "transition_reason",
                "timeout",
                "target_error_m",
                "ik_step_rad",
                "solver",
                "solver_fallback_count",
                "destination_measurement_age_s",
                "payload_force_n",
                "payload_mass_estimate_kg",
                "supported_force_n",
                "support_transfer_seconds",
                "recovery_attempts",
                "clearance_limited",
                "failed",
            ):
                if name in state:
                    value = state[name]
                    values[name] = (
                        round(value, 4) if isinstance(value, float) else value,
                        "",
                    )
            for name, (value, unit) in values.items():
                signals.append(
                    {
                        "time": round(time, 3),
                        "name": f"{trace['label']} · {name}",
                        "value": value,
                        "unit": unit,
                        "block": trace.get("signal_blocks", {}).get(
                            name, trace.get("block")
                        ),
                    }
                )
        if records:
            end = max(0.0, float(records[-1]["t"]) - offset)
            events.append(
                {
                    "time": end,
                    "label": f"{trace['label']}: {'excerpt' if trace.get('video_end_s') is not None else 'recording'} ended",
                    "block": trace.get("block"),
                }
            )
            signals.extend(
                [
                    {
                        "time": 0.0,
                        "name": f"{trace['label']} · recording",
                        "value": "active",
                        "block": trace.get("block"),
                    },
                    {
                        "time": end,
                        "name": f"{trace['label']} · recording",
                        "value": "ended (last samples retained)",
                        "block": trace.get("block"),
                    },
                ]
            )
    episode["signals"] = sorted(signals, key=lambda x: x["time"])
    episode["events"] = sorted(events, key=lambda x: x["time"])
    episode["chapters"] = sorted(chapters, key=lambda x: x["time"])
    if episode.get("multi_stage_tracks"):
        episode["chapter_tracks"] = chapter_tracks
    if episode.get("metrics_source"):
        source = episode["metrics_source"]
        evidence = json.loads(relative_file(root, source["path"]).read_text())
        runs = {
            row["id"]: row for row in evidence.get(source.get("collection", "runs"), [])
        }
        if source.get("run_id"):
            evidence = runs[source["run_id"]]
        metrics = []
        for field in source["fields"]:
            metrics.append(
                {
                    "label": field["label"],
                    "value": metric_value(evidence[field["key"]], field),
                    "unit": field.get("unit", ""),
                }
            )
            if field.get("compare_path"):
                reference = json.loads(
                    relative_file(root, field["compare_path"]).read_text()
                )[field["key"]]
                metrics[-1]["reference"] = metric_value(reference, field)
                metrics[-1]["reference_label"] = field.get("compare_label", "reference")
            if field.get("compare_run_id"):
                reference = runs[field["compare_run_id"]][field["key"]]
                metrics[-1]["reference"] = metric_value(reference, field)
                metrics[-1]["reference_label"] = field.get(
                    "compare_label", field["compare_run_id"]
                )
        episode["metrics"] = metrics


def metric_value(value, field):
    if field.get("count_where"):
        rows = value.values() if isinstance(value, dict) else value
        if not isinstance(value, (dict, list)):
            raise TypeError("Count metric requires records")
        for _ in range(field.get("count_depth", 1) - 1):
            rows = [
                child
                for row in rows
                for child in (row.values() if isinstance(row, dict) else row)
            ]
        value = sum(
            all(row[key] == expected for key, expected in field["count_where"].items())
            for row in rows
        )
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (float, int)):
        value = round(value * field.get("scale", 1), field.get("digits", 2))
        if not math.isfinite(value):
            raise ValueError("Metric must be finite")
        return value
    if not isinstance(value, str):
        raise TypeError("Metric must be scalar")
    return value


def load_policy_comparison(root: Path, path: str) -> dict:
    """Keep audit completeness separate from measured physical outcomes."""
    report = json.loads(relative_file(root, path).read_text())
    rows = []
    for run in report["runs"]:
        status = run["status"]
        if status not in {"audited", "unrun", "incomplete", "invalid"}:
            raise ValueError("Unknown comparison audit status")
        row = {"policy": run["policy"], "case": run["case"], "audit_status": status}
        if status == "audited":
            if not isinstance(run["episode_success"], bool):
                raise TypeError("Audited physical outcome must be boolean")
            row.update(
                status="success" if run["episode_success"] else "physical failure",
                seconds=run["recorded_seconds"],
                termination=run["termination_reason"],
                robots=run["robots"],
            )
        else:
            row["status"] = status
        rows.append(row)
    if len(rows) != report["expected_runs"]:
        raise ValueError("Comparison must retain every expected audit row")
    if len({(row["policy"], row["case"]) for row in rows}) != len(rows):
        raise ValueError("Duplicate policy/condition audit row")
    matched = report["complete_and_matched"] is True
    if matched and (
        any(r["audit_status"] != "audited" for r in rows) or report["matching_errors"]
    ):
        raise ValueError(
            "Comparative claim conflicts with incomplete or unmatched audit"
        )
    return {
        "path": path,
        "complete_and_matched": matched,
        "rows": rows,
        "scope": report["scope"],
        "matching_errors": report["matching_errors"],
    }


def build(source: Path = SOURCE, root: Path = RESULTS) -> Path:
    data = json.loads((source / "experiments.json").read_text())
    validate(data, root)
    for experiment in data["experiments"]:
        if experiment.get("policy_comparison_source"):
            experiment["policy_comparison"] = load_policy_comparison(
                root, experiment["policy_comparison_source"]
            )
        if experiment.get("outcomes_source"):
            evidence = json.loads(
                relative_file(root, experiment["outcomes_source"]).read_text()
            )
            outcomes = {}
            for collection in experiment.get("outcome_collections", ["runs"]):
                for run in evidence.get(collection, []):
                    outcomes[run["id"]] = {
                        "label": run["id"],
                        "split": run.get(
                            "role", experiment.get("outcome_role", "unspecified")
                        ),
                        "status": "success" if run["success"] else "incomplete",
                    }
            experiment["outcomes"] = list(outcomes.values())
            experiment["outcomes_complete"] = evidence.get("complete", True)
        for episode in experiment["episodes"]:
            enrich_episode(episode, root)
    html = (source / "index.template.html").read_text()
    payload = json.dumps(data, separators=(",", ":"), ensure_ascii=False).replace(
        "<", "\\u003c"
    )
    script = "\n".join((source / name).read_text() for name in ("graph.js", "app.js"))
    html = (
        html.replace("{{STYLE}}", (source / "lab.css").read_text())
        .replace("{{DATA}}", payload)
        .replace("{{SCRIPT}}", script)
    )
    output = root / "index.html"
    output.write_text(html)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS)
    args = parser.parse_args()
    print(build(root=args.results))
