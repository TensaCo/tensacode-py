"""Model query intents answered from an actual archived RGB-D scene receipt."""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from .mission_questions import (
    QUESTION_PROMPT,
    answer_mission_question,
    certify_question,
    decode_question,
)
from .representation import SceneGraph
from .text_language_model import MODEL_ID, MODEL_REVISION, PinnedTextLanguageModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    protocol_bytes = args.protocol.read_bytes()
    protocol = json.loads(protocol_bytes)
    if (protocol["model"], protocol["revision"], protocol["prompt"]) != (
        MODEL_ID,
        MODEL_REVISION,
        QUESTION_PROMPT,
    ):
        raise ValueError("Protocol does not match pinned model and query prompt")
    source = Path(protocol["scene_receipt"])
    source_bytes = source.read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != protocol["scene_sha256"]:
        raise ValueError("Scene receipt changed after protocol freeze")
    document = json.loads(source_bytes)
    frames = [
        f for f in document["frames"] if f["frame_index"] == protocol["frame_index"]
    ]
    if len(frames) != 1:
        raise ValueError("Protocol must identify exactly one observation frame")
    frame = frames[0]
    scene = SceneGraph.from_dict(frame["scene_graph"])
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "answers.json").exists():
        raise ValueError("Do not overwrite retained actual model receipts")
    model = PinnedTextLanguageModel(args.device)
    results = []
    for question in protocol["questions"]:
        proposal = model.propose(question["text"], QUESTION_PROMPT)
        result = {**question, "proposal": asdict(proposal)}
        try:
            request = decode_question(proposal.raw_text)
            certificate = certify_question(question["text"], request)
            result["certificate"] = certificate
            result["answer"] = (
                answer_mission_question(request, scene, frame["timestamp"])
                if certificate["status"] == "certified"
                else {"status": "needs_review"}
            )
        except (ValueError, TypeError, KeyError) as exc:
            result["answer"] = {"status": "invalid", "reason": str(exc)}
        results.append(result)
        receipt = {
            "scope": "offline model query over actual archived RGB-D observations; no robot execution",
            "historical_observed_at": frame["timestamp"],
            "scene_receipt": str(source),
            "scene_sha256": protocol["scene_sha256"],
            "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
            "model": model.provenance(),
            "questions": results,
        }
        (args.output / "answers.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(
            json.dumps({"id": question["id"], "status": result["answer"]["status"]}),
            flush=True,
        )


if __name__ == "__main__":
    main()
