"""Label-swapped likelihood review diagnostic; never an execution certificate.

This addresses boolean/explanation contradictions without letting a generated
explanation override a decision. Thresholds and reused cases are frozen before
inference. Agreement remains empirical, not proof of language fidelity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from .language_review import review_input
from .language_review_eval import CONTROLS, CampaignModel, TextCampaignModel
from .model_language import decode_request

PROMPT = """Compare the complete user request to the candidate meaning.
An exact match must preserve every object, destination, action, ordering, concurrency,
question, and constraint. A possible plan is not necessarily the requested plan.
A dropped condition, deadline, negation, speed restriction or repetition is a mismatch.
An unsupported answer matches only a request outside the available move/location
language. Unknown object references should be preserved, not replaced.
Examples: requested A then B, proposed B then A: DIFFERENT.
Requested A while B, proposed A then B: DIFFERENT.
Requested A then B, proposed only A: DIFFERENT.
Requested where parcel_0 is, proposed where parcel_1 is: DIFFERENT.
Requested A while B, proposed B while A: EQUIVALENT.
Reply with exactly one letter: {match_label} for EQUIVALENT, {different_label} for DIFFERENT.
No explanation. Use the assigned letters even if their order is unfamiliar.
"""


def decide(p_match, label_mass, threshold=0.8, min_mass=0.2):
    if len(p_match) != 2 or len(label_mass) != 2:
        raise ValueError("Both label mappings must be measured")
    if min(label_mass) < min_mass:
        return "abstain"
    if min(p_match) >= threshold:
        return "accept"
    if max(p_match) <= 1 - threshold:
        return "reject"
    return "abstain"


def paired_review(model, text, request):
    mappings = []
    for match_label, different_label in (("A", "B"), ("B", "A")):
        prompt = PROMPT.format(match_label=match_label, different_label=different_label)
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": review_input(text, request)},
        ]
        formatted = model.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = model.tokenizer(formatted, return_tensors="pt").to(model.device)
        if inputs.input_ids.shape[1] > 4096:
            raise ValueError("Review prompt exceeds bounded context")
        ids = {}
        for label in ("A", "B"):
            variants = [
                model.tokenizer.encode(s, add_special_tokens=False)
                for s in (label, " " + label)
            ]
            if any(len(v) != 1 for v in variants):
                raise ValueError("Decision labels must be single tokens")
            ids[label] = sorted({v[0] for v in variants})
        started = time.monotonic()
        with model.torch.inference_mode():
            output = model.model(**inputs, logits_to_keep=1, use_cache=False)
            probabilities = model.torch.softmax(output.logits[0, -1].float(), dim=-1)
            mass = {
                label: float(probabilities[indices].sum().cpu())
                for label, indices in ids.items()
            }
        total = mass["A"] + mass["B"]
        mappings.append(
            {
                "match_label": match_label,
                "probability_match_given_labels": mass[match_label] / max(total, 1e-30),
                "label_probability_mass": total,
                "token_ids": ids,
                "seconds": time.monotonic() - started,
                "prompt_sha256": hashlib.sha256(formatted.encode()).hexdigest(),
            }
        )
    return {
        "mappings": mappings,
        "decision": decide(
            [m["probability_match_given_labels"] for m in mappings],
            [m["label_probability_mass"] for m in mappings],
        ),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--foundation", choices=("vl2b", "text4b"), default="vl2b")
    p.add_argument("--freeze-only", action="store_true")
    a = p.parse_args()
    rows = json.loads(a.predictions.read_text())
    candidates = []
    for row in rows:
        for system, entry in row["systems"].items():
            if "invalid_output" not in entry:
                candidates.append(
                    {
                        "id": row["id"] + "/" + system,
                        "text": row["text"],
                        "raw_candidate": entry["proposal"]["raw_text"],
                        "equivalent": entry["semantic_correct"],
                        "source": "retained_actual_model_proposal",
                    }
                )
    candidates.extend(
        {
            "id": c["id"],
            "text": c["text"],
            "raw_candidate": json.dumps(c["candidate"]),
            "equivalent": c["equivalent"],
            "source": "authored_reviewer_control",
        }
        for c in CONTROLS
    )
    protocol = {
        "foundation": a.foundation,
        "scope": "Reused diagnostic cases, not held out",
        "prompt": PROMPT,
        "threshold": 0.8,
        "minimum_label_mass": 0.2,
        "candidates": candidates,
        "input_sha256": hashlib.sha256(a.predictions.read_bytes()).hexdigest(),
        "admission": "Never sufficient for physical execution; only a review diagnostic",
    }
    a.output.mkdir(parents=True, exist_ok=True)
    target = a.output / "protocol.json"
    if target.exists() and json.loads(target.read_text()) != protocol:
        raise ValueError("Protocol changed after freeze")
    target.write_text(json.dumps(protocol, indent=2) + "\n")
    if a.freeze_only:
        return
    if (a.output / "predictions.json").exists():
        raise ValueError("Do not overwrite retained predictions")
    model = (
        CampaignModel(a.device)
        if a.foundation == "vl2b"
        else TextCampaignModel(a.device)
    )
    if a.device == "cuda":
        props = model.torch.cuda.get_device_properties(0)
        (a.output / "model-runtime.json").write_text(
            json.dumps(
                {
                    "torch": model.torch.__version__,
                    "device": props.name,
                    "visible_multiprocessors": props.multi_processor_count,
                },
                indent=2,
            )
            + "\n"
        )
    outcomes = []
    for c in candidates:
        outcome = {
            **c,
            **paired_review(model, c["text"], decode_request(c["raw_candidate"])),
        }
        outcomes.append(outcome)
        (a.output / "predictions.json").write_text(
            json.dumps(outcomes, indent=2) + "\n"
        )
        print(json.dumps({"id": c["id"], "decision": outcome["decision"]}), flush=True)
    summary = {
        "count": len(outcomes),
        "correct_accepts": sum(
            r["equivalent"] and r["decision"] == "accept" for r in outcomes
        ),
        "false_accepts": [
            r["id"]
            for r in outcomes
            if not r["equivalent"] and r["decision"] == "accept"
        ],
        "correct_rejects": sum(
            not r["equivalent"] and r["decision"] == "reject" for r in outcomes
        ),
        "false_rejects": [
            r["id"] for r in outcomes if r["equivalent"] and r["decision"] == "reject"
        ],
        "abstentions": sum(r["decision"] == "abstain" for r in outcomes),
    }
    (a.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
