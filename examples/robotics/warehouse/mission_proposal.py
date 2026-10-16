"""Create a hash-bound model instruction file for the sensor-grounded executor."""

import argparse
import hashlib
import json
from pathlib import Path

from .language_repair import propose_schema_valid
from .language_review import certify_request
from .text_language_model import MODEL_ID, MODEL_REVISION, PinnedTextLanguageModel


def propose_mission(model, text, prompt, max_attempts=2):
    result = propose_schema_valid(model, text, prompt, max_attempts=max_attempts)
    receipt = {
        "text": text,
        "attempts": result["attempts"],
        "schema_valid": result["schema_valid"],
        "dispatch_authorized": False,
        "grounding": "Executor must ground this tree against fresh sensor estimates and check capabilities/resources/motion feasibility",
    }
    if not result["schema_valid"]:
        return {**receipt, "status": "invalid"}, None
    request = result["request"]
    certificate = certify_request(text, request)
    receipt["certificate"] = certificate
    if certificate["status"] != "certified" or request["kind"] != "instruction":
        return {**receipt, "status": "needs_review"}, None
    return {**receipt, "status": "certified_instruction_proposal"}, request[
        "tree"
    ].to_dict()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    protocol_bytes = args.protocol.read_bytes()
    protocol = json.loads(protocol_bytes)
    if (protocol["model"], protocol["revision"]) != (MODEL_ID, MODEL_REVISION):
        raise ValueError("Protocol requires this explicitly pinned foundation")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "proposal.json").exists() or (
        args.output / "instruction.json"
    ).exists():
        raise ValueError("Refusing to overwrite an earlier model proposal")
    model = PinnedTextLanguageModel(args.device)
    receipt, instruction = propose_mission(
        model, protocol["text"], protocol["prompt"], protocol["max_attempts"]
    )
    receipt.update(
        model=model.provenance(),
        protocol_sha256=hashlib.sha256(protocol_bytes).hexdigest(),
    )
    if instruction is not None:
        serialized = json.dumps(instruction, indent=2) + "\n"
        (args.output / "instruction.json").write_text(serialized)
        receipt["instruction_sha256"] = hashlib.sha256(serialized.encode()).hexdigest()
    (args.output / "proposal.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "instruction_written": instruction is not None,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
