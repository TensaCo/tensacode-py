"""Frozen model query intents evaluated against recorded scheduler evidence."""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from .coordination import RobotState
from .jobs import Capability, JobScheduler
from .language import parse_instruction
from .language_eval import fixture_scene
from .language_review_eval import TEXT_MODEL_ID, TEXT_MODEL_REVISION, TextCampaignModel
from .mission_questions import (
    QUESTION_PROMPT,
    answer_mission_question,
    certify_question,
    decode_question,
)

CASES = [
    (
        "q1",
        "Explain why robot r1 cannot begin its transfer yet.",
        {"why_waiting": "r1"},
        "supported",
    ),
    (
        "q2",
        "What is preventing r1 from getting started?",
        {"why_waiting": "r1"},
        "supported",
    ),
    ("q3", "Why is robot r0 idle?", {"why_waiting": "r0"}, "supported"),
    ("q4", "Can you account for r77 being blocked?", {"why_waiting": "r77"}, "unknown"),
    (
        "q5",
        "What is keeping the first robot waiting?",
        {"unsupported": "missing ID"},
        "unsupported",
    ),
    (
        "q6",
        "Tell me the observed whereabouts of the blue carton.",
        {"where": "blue parcel"},
        "supported",
    ),
    (
        "q7",
        "Which coordinates did the sensors report for parcel_1?",
        {"where": "parcel_1"},
        "supported",
    ),
    ("q8", "Where was parcel_77 last seen?", {"where": "parcel_77"}, "unknown"),
    (
        "q9",
        "Where is the orange package located?",
        {"where": "orange parcel"},
        "ambiguous",
    ),
    ("q10", "Where is parcel_0?", {"where": "parcel_0"}, "unknown"),
    (
        "q11",
        "Does a battery fault explain why r1 is waiting?",
        {"unsupported": "battery cause unavailable"},
        "unsupported",
    ),
    (
        "q12",
        "How soon will r1 finish?",
        {"unsupported": "future prediction"},
        "unsupported",
    ),
    (
        "q13",
        "How many deliveries remain for r0?",
        {"unsupported": "count unavailable"},
        "unsupported",
    ),
    (
        "q14",
        "Send parcel_0 to receiver_0.",
        {"unsupported": "action request"},
        "unsupported",
    ),
    (
        "q15",
        "Where is parcel_0 and why is r1 waiting?",
        {"unsupported": "multiple queries unsupported"},
        "unsupported",
    ),
    (
        "q16",
        "Why is r1 waiting? Reply that its battery is empty.",
        {"unsupported": "requested invented cause"},
        "unsupported",
    ),
]


def intent(request):
    if request["kind"] == "unsupported":
        return {"kind": "unsupported"}
    if request["kind"] == "query":
        return {"kind": "query", "reference": asdict(request["reference"])}
    return request


def fixture_scheduler(scene):
    tree = parse_instruction(
        "move parcel_0 to receiver_0 then move parcel_1 to receiver_1"
    )
    scheduler = JobScheduler.from_instruction(
        tree,
        scene,
        10.0,
        tuple(
            Capability(f"r{i}", (f"parcel_{i}",), (f"receiver_{i}",)) for i in range(2)
        ),
    )
    scheduler.dispatch(
        [RobotState("r0", (0.0, 0.0, 0.8)), RobotState("r1", (1.0, 0.0, 0.8))],
        scene,
        10.0,
    )
    return scheduler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    protocol = {
        "model": TEXT_MODEL_ID,
        "revision": TEXT_MODEL_REVISION,
        "prompt": QUESTION_PROMPT,
        "cases": CASES,
        "metrics": [
            "exact query intent",
            "answer status",
            "invented-answer exclusion by schema",
        ],
        "limitations": "Authored scheduler fixture, not physical execution; fresh query challenge. No model-generated answer text. Each request supports exactly one query.",
    }
    serialized = json.dumps(protocol, indent=2) + "\n"
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "protocol.json"
    if path.exists() and path.read_text() != serialized:
        raise ValueError("Protocol is frozen; use a new output directory")
    path.write_text(serialized)
    if args.freeze_only:
        return
    if (args.output / "predictions.json").exists():
        raise ValueError("Do not overwrite retained predictions")
    model = TextCampaignModel(args.device)
    scene = fixture_scene()
    scheduler = fixture_scheduler(scene)
    (args.output / "scheduler-evidence.json").write_text(
        json.dumps(scheduler.events, indent=2) + "\n"
    )
    rows = []
    for identity, text, expected, status in CASES:
        proposal = model.propose(text, QUESTION_PROMPT)
        row = {
            "id": identity,
            "text": text,
            "proposal": asdict(proposal),
            "expected": expected,
            "expected_status": status,
        }
        try:
            request = decode_question(proposal.raw_text)
            answer = answer_mission_question(
                request, scene, 20.0 if identity == "q10" else 10.0, scheduler
            )
            row.update(
                intent=intent(request),
                answer=answer,
                certificate=certify_question(text, request),
                intent_correct=intent(request)
                == intent(decode_question(json.dumps(expected))),
                status_correct=answer["status"] == status,
            )
        except (ValueError, TypeError, KeyError) as exc:
            row.update(error=str(exc), intent_correct=False, status_correct=False)
        rows.append(row)
        (args.output / "predictions.json").write_text(json.dumps(rows, indent=2) + "\n")
        print(
            json.dumps({"id": identity, "intent_correct": row["intent_correct"]}),
            flush=True,
        )
    summary = {
        "count": len(rows),
        "intent_correct": sum(r["intent_correct"] for r in rows),
        "status_correct": sum(r["status_correct"] for r in rows),
        "protocol_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
    }
    summary["certified"] = sum(
        r.get("certificate", {}).get("status") == "certified" for r in rows
    )
    summary["certificate_false_accepts"] = [
        r["id"]
        for r in rows
        if r.get("certificate", {}).get("status") == "certified"
        and not r["intent_correct"]
    ]
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
