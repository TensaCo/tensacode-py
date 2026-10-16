"""Run a predeclared JSON challenge without replacing any model proposal."""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from .language_eval import fixture_scene
from .language_repair import propose_schema_valid
from .language_review import (
    UncertifiedLanguage,
    certificate_parse,
    certify_request,
    signature,
)
from .language_review_eval import (
    MODEL_ID,
    MODEL_REVISION,
    TEXT_MODEL_ID,
    TEXT_MODEL_REVISION,
    CampaignModel,
    TextCampaignModel,
)
from .model_language import ModelProposal, decode_request, ground_request


def run(protocol, output, device):
    """The JSON protocol contains only requests, expected meanings and prompt text."""
    model_key = (protocol["model"], protocol["revision"])
    foundations = {
        (MODEL_ID, MODEL_REVISION): CampaignModel,
        (TEXT_MODEL_ID, TEXT_MODEL_REVISION): TextCampaignModel,
    }
    if model_key not in foundations:
        raise ValueError("Challenge requires an explicitly supported pinned model")
    if not isinstance(protocol["prompt"], str) or not protocol["cases"]:
        raise ValueError("Protocol needs a prompt and cases")
    # Validate all gold meanings before any model load or inference.
    for case in protocol["cases"]:
        decode_request(json.dumps(case["expected"]))
    initial = {}
    if "initial_predictions" in protocol:
        binding = protocol["initial_predictions"]
        retained = Path(binding["path"]).read_bytes()
        if hashlib.sha256(retained).hexdigest() != binding["sha256"]:
            raise ValueError("Retained first-pass predictions changed")
        initial = {row["id"]: row for row in json.loads(retained)}
        for case in protocol["cases"]:
            if initial[case["id"]]["text"] != case["text"]:
                raise ValueError("Initial proposal does not match original request")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "predictions.json").exists():
        raise ValueError("Refusing to overwrite retained predictions")
    frozen = json.dumps(protocol, indent=2) + "\n"
    target = output / "protocol.json"
    if target.exists() and target.read_text() != frozen:
        raise ValueError("Output protocol differs from supplied frozen protocol")
    target.write_text(frozen)
    model = foundations[model_key](device)
    runtime = {"device": device, "torch": model.torch.__version__}
    if device == "cuda":
        properties = model.torch.cuda.get_device_properties(0)
        runtime.update(
            device_name=properties.name,
            visible_multiprocessors=properties.multi_processor_count,
        )
    (output / "model-runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    rows = []
    for case in protocol["cases"]:
        expected = signature(decode_request(json.dumps(case["expected"])))
        if "schema_repair_attempts" in protocol:
            retained = (
                ModelProposal(**initial[case["id"]]["proposal"]) if initial else None
            )
            result = propose_schema_valid(
                model,
                case["text"],
                protocol["prompt"],
                max_attempts=protocol["schema_repair_attempts"],
                initial_proposal=retained,
            )
            proposal = ModelProposal(**result["attempts"][-1]["proposal"])
            attempts = result["attempts"]
        else:
            proposal = model.propose(case["text"], protocol["prompt"])
            attempts = [{"proposal": asdict(proposal)}]
        row = {**case, "proposal": asdict(proposal), "attempts": attempts}
        try:
            row["first_pass_correct"] = (
                signature(decode_request(attempts[0]["proposal"]["raw_text"]))
                == expected
            )
        except (ValueError, TypeError, KeyError):
            row["first_pass_correct"] = False
        try:
            request = decode_request(proposal.raw_text)
            row.update(
                semantic_correct=signature(request) == expected,
                candidate_signature=signature(request),
                certificate=certify_request(case["text"], request),
                grounding=ground_request(
                    request, fixture_scene(), case.get("now", 10.0)
                ),
            )
        except (ValueError, TypeError, KeyError) as exc:
            row.update(semantic_correct=False, error=str(exc))
        try:
            authored = certificate_parse(case["text"])
            row["authored_grammar"] = {
                "in_grammar": True,
                "semantic_correct": signature(authored) == expected,
            }
        except UncertifiedLanguage:
            row["authored_grammar"] = {"in_grammar": False, "semantic_correct": False}
        rows.append(row)
        (output / "predictions.json").write_text(json.dumps(rows, indent=2) + "\n")
        print(
            json.dumps({"id": case["id"], "semantic_correct": row["semantic_correct"]}),
            flush=True,
        )
    certified = [
        r for r in rows if r.get("certificate", {}).get("status") == "certified"
    ]
    summary = {
        "count": len(rows),
        "semantic_correct": sum(r["semantic_correct"] for r in rows),
        "first_pass_correct": sum(r["first_pass_correct"] for r in rows),
        "repair_invocations": sum(len(r["attempts"]) > 1 for r in rows),
        "certified": len(certified),
        "certificate_false_accepts": [
            r["id"] for r in certified if not r["semantic_correct"]
        ],
        "authored_grammar_correct": sum(
            r["authored_grammar"]["semantic_correct"] for r in rows
        ),
        "protocol_sha256": hashlib.sha256(frozen.encode()).hexdigest(),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    run(json.loads(args.protocol.read_text()), args.output, args.device)


if __name__ == "__main__":
    main()
