"""Predeclared v3 parser/reviewer/certificate experiment; every model output retained."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

from .language_eval import fixture_scene
from .language_review import (
    REVIEW_PROMPT,
    V3_PROMPT,
    certify_request,
    decode_review,
    review_input,
    signature,
)
from .model_language import (
    MODEL_ID,
    MODEL_REVISION,
    V2_SYSTEM_PROMPT,
    FrozenLanguageModel,
    ModelProposal,
    decode_request,
    ground_request,
)
from .text_language_model import (
    MODEL_ID as TEXT_MODEL_ID,
)
from .text_language_model import (
    MODEL_REVISION as TEXT_MODEL_REVISION,
)
from .text_language_model import (
    PinnedTextLanguageModel as TextCampaignModel,
)


def move(obj, dest):
    return ["move", obj, dest]


def case(identity, text, expected, status="supported", split="fresh"):
    return {
        "id": identity,
        "text": text,
        "expected": expected,
        "grounding_status": status,
        "split": split,
    }


A = move("parcel_0", "receiver_0")
B = move("parcel_1", "receiver_1")
C = move("parcel_2", "receiver_1")
CASES = [
    case(
        "d1",
        "Deliver parcel_0 to receiver_0, and once that is done, deliver parcel_2 to receiver_1.",
        {"instruction": ["sequence", A, C]},
        split="development",
    ),
    case(
        "d2",
        "Move parcel_0 to receiver_0 unless the destination is occupied.",
        {"unsupported": "condition"},
        "unsupported",
        "development",
    ),
    case(
        "d3",
        "Put parcel_0 on receiver_0 in less than two seconds.",
        {"unsupported": "deadline"},
        "unsupported",
        "development",
    ),
    case(
        "d4",
        "Please handle both deliveries concurrently: parcel_0 goes to receiver_0, and parcel_1 goes to receiver_1.",
        {"instruction": ["parallel", A, B]},
        split="development",
    ),
    case(
        "d5",
        "Before bringing parcel_1 to receiver_1, put parcel_0 on receiver_0.",
        {"instruction": ["sequence", A, B]},
        split="development",
    ),
    case(
        "d6",
        "Where did you last observe the blue package?",
        {"where": "blue parcel"},
        split="development",
    ),
    case(
        "d7",
        "Move the orange parcel to receiver_0.",
        {"instruction": move("orange parcel", "receiver_0")},
        "ambiguous",
        "development",
    ),
    case(
        "d8",
        "Take it over there.",
        {"unsupported": "missing references"},
        "unsupported",
        "development",
    ),
    case(
        "f1",
        "Bring parcel_1 to receiver_0 while carrying parcel_0 to receiver_1.",
        {
            "instruction": [
                "parallel",
                move("parcel_1", "receiver_0"),
                move("parcel_0", "receiver_1"),
            ]
        },
    ),
    case(
        "f2",
        "After moving parcel_1 to receiver_1, take parcel_0 to receiver_0.",
        {"instruction": ["sequence", B, A]},
    ),
    case(
        "f3",
        "Take parcel_0 to receiver_0; afterwards bring parcel_2 to receiver_1.",
        {"instruction": ["sequence", A, C]},
    ),
    case(
        "f4",
        "Move parcel_0 to receiver_0 while moving parcel_1 to receiver_1; then carry parcel_2 to receiver_0.",
        {
            "instruction": [
                "sequence",
                ["parallel", A, B],
                move("parcel_2", "receiver_0"),
            ]
        },
    ),
    case(
        "f5",
        "Before taking parcel_2 to receiver_1, move parcel_0 to receiver_0 while carrying parcel_1 to receiver_1.",
        {"instruction": ["sequence", ["parallel", A, B], C]},
    ),
    case(
        "f6",
        "Move the parcel left of parcel_1 to receiver_1.",
        {"instruction": move("parcel left of parcel_1", "receiver_1")},
    ),
    case(
        "f7",
        "Could you put the blue carton onto receiver_1?",
        {"instruction": move("blue parcel", "receiver_1")},
    ),
    case("f8", "Where did you last observe parcel_2?", {"where": "parcel_2"}),
    case(
        "f9",
        "Tell me where the package left of parcel_1 is.",
        {"where": "parcel left of parcel_1"},
    ),
    case(
        "f10",
        "Carry the orange parcel to receiver_0.",
        {"instruction": move("orange parcel", "receiver_0")},
        "ambiguous",
    ),
    case(
        "f11",
        "Take parcel_8 to receiver_0.",
        {"instruction": move("parcel_8", "receiver_0")},
        "unknown",
    ),
    case(
        "f12",
        "Move parcel_0 to receiver_0 provided that no robot is nearby.",
        {"unsupported": "condition"},
        "unsupported",
    ),
    case(
        "f13",
        "Do not deliver parcel_0 to receiver_1.",
        {"unsupported": "negation"},
        "unsupported",
    ),
    case(
        "f14",
        "After 30 seconds, move parcel_0 to receiver_0.",
        {"unsupported": "delay"},
        "unsupported",
    ),
    case(
        "f15",
        "Repeat moving parcel_0 to receiver_0 three times.",
        {"unsupported": "repetition"},
        "unsupported",
    ),
    case(
        "f16",
        "How heavy is the blue package?",
        {"unsupported": "weight query"},
        "unsupported",
    ),
]

# Authored perturbation controls evaluate the REVIEWER only. Never substituted for
# generated candidates in parser scoring or execution admission.
CONTROLS = [
    {
        "id": "r1",
        "text": "Move parcel_0 to receiver_0 while moving parcel_1 to receiver_1.",
        "candidate": {"instruction": ["sequence", A, B]},
        "equivalent": False,
    },
    {
        "id": "r2",
        "text": "Move parcel_0 to receiver_0, then move parcel_1 to receiver_1.",
        "candidate": {"instruction": ["parallel", A, B]},
        "equivalent": False,
    },
    {
        "id": "r3",
        "text": "Move parcel_0 to receiver_0, then move parcel_1 to receiver_1.",
        "candidate": {"instruction": ["sequence", B, A]},
        "equivalent": False,
    },
    {
        "id": "r4",
        "text": "Move parcel_0 to receiver_0, then move parcel_1 to receiver_1.",
        "candidate": {"instruction": A},
        "equivalent": False,
    },
    {
        "id": "r5",
        "text": "Do not move parcel_0 to receiver_0.",
        "candidate": {"instruction": A},
        "equivalent": False,
    },
    {
        "id": "r6",
        "text": "Move parcel_0 to receiver_0 if the station is empty.",
        "candidate": {"instruction": A},
        "equivalent": False,
    },
    {
        "id": "r7",
        "text": "Where is parcel_0?",
        "candidate": {"where": "parcel_1"},
        "equivalent": False,
    },
    {
        "id": "r8",
        "text": "Move parcel_0 to receiver_0.",
        "candidate": {"instruction": move("parcel_0", "receiver_1")},
        "equivalent": False,
    },
    {
        "id": "r9",
        "text": "Move parcel_0 to receiver_0 while moving parcel_1 to receiver_1.",
        "candidate": {"instruction": ["parallel", B, A]},
        "equivalent": True,
    },
    {
        "id": "r10",
        "text": "Where is parcel_9?",
        "candidate": {"where": "parcel_9"},
        "equivalent": True,
    },
]


class CampaignModel(FrozenLanguageModel):
    def __init__(self, device):
        super().__init__(threads=2)
        if device not in {"cpu", "cuda"}:
            raise ValueError("Only explicitly allocated CPU/CUDA devices are supported")
        self.device = device
        self.model.to(device)

    def propose(self, instruction, system_prompt):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instruction},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        if inputs.input_ids.shape[1] > 4096:
            raise ValueError("Experiment prompt exceeds 4096 token bound")
        started = time.monotonic()
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=384,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = output[0, inputs.input_ids.shape[1] :]
        return ModelProposal(
            self.tokenizer.decode(generated, skip_special_tokens=True).strip(),
            int(generated.numel()),
            time.monotonic() - started,
            hashlib.sha256(text.encode()).hexdigest(),
        )


def review(model, text, request):
    proposal = model.propose(review_input(text, request), REVIEW_PROMPT)
    try:
        result = decode_review(proposal.raw_text)
    except ValueError as exc:
        result = {"equivalent": False, "invalid": str(exc)}
    return {"proposal": asdict(proposal), "result": result}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--freeze-only", action="store_true")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--foundation", choices=("vl2b", "text4b"), default="vl2b")
    a = p.parse_args()
    cases = (
        CASES
        if a.foundation == "vl2b"
        else [
            {**c, "original_split": c["split"], "split": "comparison_reuse"}
            for c in CASES
        ]
    )
    model_id = MODEL_ID if a.foundation == "vl2b" else TEXT_MODEL_ID
    revision = MODEL_REVISION if a.foundation == "vl2b" else TEXT_MODEL_REVISION
    a.output.mkdir(parents=True, exist_ok=True)
    protocol = {
        "model": model_id,
        "revision": revision,
        "cases": cases,
        "review_controls": CONTROLS,
        "prompts": {
            "baseline": V2_SYSTEM_PROMPT,
            "improved": V3_PROMPT,
            "review": REVIEW_PROMPT,
        },
        "predeclared_metrics": [
            "exact semantic signature accuracy",
            "observed-scene grounding status",
            "neural review false acceptance and coverage",
            "full-grammar certificate false acceptance and coverage",
            "review-control accuracy",
        ],
        "admission": "Only complete grammar certificate + fresh unambiguous grounding permits physical dispatch; neural review is diagnostic, never sufficient",
        "limitations": [
            "Authored small challenge",
            "Same frozen model parses and reviews; correlated errors possible",
            "No supported wait, condition, deadline, or repeat-count executor operations",
            "Authored fixture scene isolates language, not physical performance",
        ],
    }
    target = a.output / "protocol.json"
    if target.exists() and json.loads(target.read_text()) != protocol:
        raise ValueError(
            "Protocol changed after freezing; use a distinct run directory"
        )
    target.write_text(json.dumps(protocol, indent=2) + "\n")
    if a.freeze_only:
        return
    if (a.output / "predictions.json").exists():
        raise ValueError("Never overwrite retained model predictions")
    model = (
        CampaignModel(a.device)
        if a.foundation == "vl2b"
        else TextCampaignModel(a.device)
    )
    runtime = {
        "device": a.device,
        "model": model_id,
        "revision": revision,
        "torch": model.torch.__version__,
    }
    if a.device == "cuda":
        properties = model.torch.cuda.get_device_properties(0)
        runtime.update(
            device_name=properties.name,
            visible_multiprocessors=properties.multi_processor_count,
        )
    (a.output / "model-runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    scene = fixture_scene()
    rows = []
    for item in cases:
        expected = signature(decode_request(json.dumps(item["expected"])))
        row = {**item, "systems": {}}
        for label, prompt in (("baseline", V2_SYSTEM_PROMPT), ("improved", V3_PROMPT)):
            proposal = model.propose(item["text"], prompt)
            entry = {"proposal": asdict(proposal)}
            try:
                request = decode_request(proposal.raw_text)
                entry.update(
                    signature=signature(request),
                    semantic_correct=signature(request) == expected,
                    grounding=ground_request(request, scene, 10.0),
                    certificate=certify_request(item["text"], request),
                )
                entry["review"] = review(model, item["text"], request)
            except (ValueError, TypeError, KeyError) as exc:
                entry.update(
                    semantic_correct=False,
                    invalid_output=str(exc),
                    certificate={"status": "abstain"},
                )
            row["systems"][label] = entry
        rows.append(row)
        (a.output / "predictions.json").write_text(json.dumps(rows, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "id": item["id"],
                    **{k: v["semantic_correct"] for k, v in row["systems"].items()},
                }
            ),
            flush=True,
        )
    controls = []
    for item in CONTROLS:
        response = review(
            model, item["text"], decode_request(json.dumps(item["candidate"]))
        )
        controls.append(
            {
                **item,
                **response,
                "correct": response["result"]["equivalent"] == item["equivalent"],
            }
        )
        (a.output / "review-controls.json").write_text(
            json.dumps(controls, indent=2) + "\n"
        )
    summary = {
        "results": {},
        "review_controls": {
            "correct": sum(r["correct"] for r in controls),
            "count": len(controls),
            "false_accepts": [
                r["id"]
                for r in controls
                if not r["equivalent"] and r["result"]["equivalent"]
            ],
        },
    }
    for split in sorted({r["split"] for r in rows}):
        subset = [r for r in rows if r["split"] == split]
        summary["results"][split] = {}
        for label in ("baseline", "improved"):
            entries = [r["systems"][label] for r in subset]
            summary["results"][split][label] = {
                "count": len(entries),
                "semantic_correct": sum(e["semantic_correct"] for e in entries),
                "neural_review_accepts": sum(
                    e.get("review", {}).get("result", {}).get("equivalent", False)
                    for e in entries
                ),
                "neural_review_false_accepts": sum(
                    not e["semantic_correct"]
                    and e.get("review", {}).get("result", {}).get("equivalent", False)
                    for e in entries
                ),
                "certified": sum(
                    e["certificate"]["status"] == "certified" for e in entries
                ),
                "certificate_false_accepts": sum(
                    not e["semantic_correct"]
                    and e["certificate"]["status"] == "certified"
                    for e in entries
                ),
            }
    (a.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
