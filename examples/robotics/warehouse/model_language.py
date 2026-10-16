"""Optional frozen-foundation parser; generated text never executes as code.

The model only proposes a typed instruction tree. Reference resolution, admission,
and question answering are deterministic operations on observed scene evidence.
No model weights are trained by this example. Importing this module loads no model.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
MODEL_REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"


@dataclass(frozen=True)
class ModelProposal:
    raw_text: str
    generated_tokens: int
    seconds: float
    prompt_sha256: str


def strict_json(text: str) -> dict[str, Any]:
    """Accept a single JSON object, without code fences or silent repair."""
    if len(text) > 32000:
        raise ValueError("Model response exceeds bounded JSON size")

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate model key: {key}")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")

    try:
        value = json.loads(
            text, object_pairs_hook=pairs, parse_constant=invalid_constant
        )
    except RecursionError as exc:
        raise ValueError("Model JSON nesting is excessive") from exc
    if not isinstance(value, dict):
        raise ValueError("Model proposal must be a JSON object")  # noqa: TRY004 - JSON validation boundary
    return value


class FrozenLanguageModel:
    """Pinned local text-only Qwen inference, CPU only and bounded generation."""

    def __init__(self, *, threads: int = 2):
        import torch
        from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration

        if threads not in (1, 2):
            raise ValueError("This example allows one or two CPU threads")
        torch.set_num_threads(threads)
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            local_files_only=True,
        )
        self.model = (
            Qwen3VLForConditionalGeneration.from_pretrained(
                MODEL_ID,
                revision=MODEL_REVISION,
                local_files_only=True,
                dtype=torch.bfloat16,
                attn_implementation="eager",
            )
            .to("cpu")
            .eval()
        )

    def propose(self, instruction: str, system_prompt: str) -> ModelProposal:
        if not isinstance(instruction, str) or not 0 < len(instruction) <= 1500:
            raise ValueError("Instruction must contain 1–1500 characters")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instruction},
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt")
        if inputs.input_ids.shape[1] > 3072:
            raise ValueError("Prompt exceeds 3072 token limit")
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
            raw_text=self.tokenizer.decode(generated, skip_special_tokens=True).strip(),
            generated_tokens=int(generated.numel()),
            seconds=time.monotonic() - started,
            prompt_sha256=hashlib.sha256(text.encode()).hexdigest(),
        )


SYSTEM_PROMPT = """You translate warehouse language into a strict, data-only syntax tree.
Return ONLY one JSON object, with no markdown or commentary.
Supported operation: picking up and placing a parcel at a receiving station.
Normalize ordinary box/package/item/carton references to generic name "parcel";
normalize tray/receiving platform/destination references to generic name "station".
Keep explicit identifiers (such as parcel_0, receiver_1, dock_a) unchanged, lowercased.
A reference is {"name":string,"color":string|null,"relation":string|null,"anchor":reference|null}.
Allowed colors: red, green, blue, yellow, magenta, orange, white, black.
Allowed relations: "in", "on", "left of", "right of". A relation requires an anchor.
A leaf instruction is {"operation":"pick_place","object":reference,"destination":reference,"children":[]}.
Sequence instructions use operation "sequence", object:null, destination:null,
children:[instruction,...], in execution order. For explicitly simultaneous actions
use "parallel" instead of "sequence". Preserve all requested clauses and ordering.
An instruction response is {"kind":"instruction","tree":instruction}.
Location questions use {"kind":"query","reference":reference}. Do not answer them;
the observed scene database will answer later. Do not invent positions or objects.
Unsupported capabilities, vague pronouns without an antecedent, missing destinations,
or conditions that cannot be represented use {"kind":"unsupported","reason":string}.
Never discard a negation, timing constraint, speed constraint, or safety constraint.
If such a constraint cannot be represented, mark the entire request unsupported.
Do not choose one object when the reference could identify several. Preserve the
reference as given; the downstream observed-scene resolver handles ambiguity.
Examples:
"Please take the blue package to receiver_0."
{"kind":"instruction","tree":{"operation":"pick_place","object":{"name":"parcel","color":"blue","relation":null,"anchor":null},"destination":{"name":"receiver_0","color":null,"relation":null,"anchor":null},"children":[]}}
"Can you tell me the location of parcel_1?"
{"kind":"query","reference":{"name":"parcel_1","color":null,"relation":null,"anchor":null}}
"Charge your battery."
{"kind":"unsupported","reason":"Charging is not a supported operation."}
"Move parcel_0 to receiver_0 without entering the aisle."
{"kind":"unsupported","reason":"The route exclusion constraint is not representable."}
"Do not move parcel_0."
{"kind":"unsupported","reason":"A negative instruction is not an executable pick/place request."}
"Place parcel_0 at receiver_0, then place parcel_1 at receiver_1."
{"kind":"instruction","tree":{"operation":"sequence","object":null,"destination":null,"children":[{"operation":"pick_place","object":{"name":"parcel_0","color":null,"relation":null,"anchor":null},"destination":{"name":"receiver_0","color":null,"relation":null,"anchor":null},"children":[]},{"operation":"pick_place","object":{"name":"parcel_1","color":null,"relation":null,"anchor":null},"destination":{"name":"receiver_1","color":null,"relation":null,"anchor":null},"children":[]}]}}
"""


# Development v2: fewer generated tokens, explicit compositional exemplars.
# The alias normalization and AST expansion below are authored, not learned.
V2_SYSTEM_PROMPT = """Translate warehouse requests into concise JSON. Output ONLY JSON.
Supported abilities: move parcels to stations, compose multiple moves sequentially
or simultaneously, and ask for the observed location of an object. Never answer
questions yourself. No counting, weight questions, navigation restrictions,
conditions, negation, deadlines, speed constraints, charging, or other abilities.

Use exactly one of these forms:
{"instruction":["move","OBJECT REFERENCE","DESTINATION REFERENCE"]}
{"instruction":["parallel",MOVE,MOVE,...]}
{"instruction":["sequence",MOVE,MOVE,...]}
{"where":"OBJECT REFERENCE"}
{"unsupported":"brief reason"}
MOVE is an array ["move","object reference","destination reference"].
Reference strings may contain an explicit ID, a color plus parcel/station, or a
relation left of/right of/in/on another reference. Keep explicit IDs unchanged.
Box/carton/package/item means parcel. Tray/platform means station.
Colors: red, green, blue, yellow, magenta, orange, white, black. Other colors unsupported.
Keep every requested action. Parallel means while/concurrently/simultaneously/at
the same time. Sequence means then/afterwards/before/after, in execution order.
Ambiguous references stay ambiguous; do not choose a specific ID. Missing
references or unsupported constraints make the whole request unsupported.

Examples:
User: Please relocate the blue carton onto receiver_1.
{"instruction":["move","blue parcel","receiver_1"]}
User: Deliver parcel_0 to receiver_0 and at the same time take parcel_1 to receiver_1.
{"instruction":["parallel",["move","parcel_0","receiver_0"],["move","parcel_1","receiver_1"]]}
User: Move parcel_0 to receiver_0 while moving parcel_1 to receiver_1.
{"instruction":["parallel",["move","parcel_0","receiver_0"],["move","parcel_1","receiver_1"]]}
User: Before bringing parcel_1 to receiver_1, put parcel_0 on receiver_0.
{"instruction":["sequence",["move","parcel_0","receiver_0"],["move","parcel_1","receiver_1"]]}
User: Move parcel_1 to receiver_0, then move parcel_0 to receiver_1.
{"instruction":["sequence",["move","parcel_1","receiver_0"],["move","parcel_0","receiver_1"]]}
User: Where did you last observe the blue package?
{"where":"blue parcel"}
User: Tell me the location of the parcel left of parcel_1.
{"where":"parcel left of parcel_1"}
User: Take the package to the left of parcel_1 to receiver_0.
{"instruction":["move","parcel left of parcel_1","receiver_0"]}
User: Move the orange parcel to receiver_0.
{"instruction":["move","orange parcel","receiver_0"]}
User: Where is parcel_9?
{"where":"parcel_9"}
User: Send the purple parcel to receiver_0.
{"unsupported":"Unsupported color"}
User: Do not move parcel_0 to receiver_0.
{"unsupported":"Negative instruction"}
User: Carry parcel_0 to receiver_0 only if the aisle is empty.
{"unsupported":"Conditional instruction"}
User: Take it over there.
{"unsupported":"Missing object and destination references"}
User: How much does parcel_0 weigh?
{"unsupported":"Only location questions are supported"}
"""


def _concise_reference(text):
    import re

    from .language import parse_reference

    if not isinstance(text, str) or not 0 < len(text) <= 300:
        raise ValueError("Concise references require 1–300 characters")
    aliases = {
        "box": "parcel",
        "carton": "parcel",
        "package": "parcel",
        "item": "parcel",
        "tray": "station",
        "platform": "station",
    }
    # Word boundaries preserve identifiers such as box_0 exactly.
    text = re.sub(
        r"\b(?:box|carton|package|item|tray|platform)\b",
        lambda match: aliases[match.group()],
        text.lower(),
    )
    text = re.sub(r"\bto the (left|right) of\b", r"\1 of", text)
    return parse_reference(text)


def _concise_instruction(value, depth=0):
    from .language import Instruction

    if depth > 16 or not isinstance(value, list) or not value:
        raise ValueError("Invalid concise instruction tree")
    if value[0] == "move" and len(value) == 3:
        return Instruction(
            "pick_place", _concise_reference(value[1]), _concise_reference(value[2])
        )
    if value[0] in ("sequence", "parallel") and 3 <= len(value) <= 65:
        return Instruction(
            value[0],
            children=tuple(_concise_instruction(v, depth + 1) for v in value[1:]),
        )
    raise ValueError("Invalid concise instruction operation or arity")


def decode_request(text: str) -> dict[str, Any]:
    """Validate every model field before admitting an instruction or question."""
    from dataclasses import asdict

    from .language import instruction_from_dict, reference_from_dict

    value = strict_json(text)
    if set(value) == {"instruction"}:
        return {
            "kind": "instruction",
            "tree": _concise_instruction(value["instruction"]),
        }
    if set(value) == {"where"}:
        return {
            "kind": "query",
            "reference": reference_from_dict(
                asdict(_concise_reference(value["where"]))
            ),
        }
    if set(value) == {"unsupported"}:
        value = {"kind": "unsupported", "reason": value["unsupported"]}
    kind = value.get("kind")
    if kind == "instruction" and set(value) == {"kind", "tree"}:
        return {"kind": kind, "tree": instruction_from_dict(value["tree"])}
    if kind == "query" and set(value) == {"kind", "reference"}:
        return {"kind": kind, "reference": reference_from_dict(value["reference"])}
    if (
        kind == "unsupported"
        and set(value) == {"kind", "reason"}
        and isinstance(value["reason"], str)
        and 0 < len(value["reason"]) <= 1000
    ):
        return value
    raise ValueError("Unknown or malformed model request envelope")


def ground_request(
    request: dict[str, Any], scene, now: float, max_age: float = 2.0
) -> dict:
    """Model-generated questions receive answers exclusively from observations."""
    from dataclasses import asdict

    from .language import GroundingError, ground_instruction, resolve

    try:
        if request["kind"] == "unsupported":
            return {"status": "unsupported", "reason": request["reason"]}
        if request["kind"] == "instruction":
            tasks = ground_instruction(request["tree"], scene, now, max_age)
            return {
                "status": "supported",
                "tasks": [asdict(task) for task in tasks],
                "scene_revision": scene.revision,
            }
        if request["kind"] != "query":
            raise ValueError(
                "Only validated instruction, query, or unsupported requests are accepted"
            )
        entity = resolve(request["reference"], scene, now, max_age)
        return {
            "status": "supported",
            "entity_ids": [entity.id],
            "sources": [entity.source],
            "scene_revision": scene.revision,
            "position_m": list(entity.position),
            "age_s": now - entity.observed_at,
            "uncertainty_m": entity.uncertainty_m,
            "confidence": entity.confidence,
            "answer": f"{entity.id} was observed at {tuple(round(v, 3) for v in entity.position)} m in the warehouse frame.",
        }
    except GroundingError as exc:
        return {
            "status": exc.status,
            "reason": str(exc),
            "candidates": list(exc.candidates),
            "scene_revision": scene.revision,
        }


def check_semantic_admission(instruction: str, request: dict[str, Any]) -> dict:
    """Authored conservative cue checks, NOT a proof of full language fidelity.

    Never rewrites a model tree. Mixed sequence/concurrency requests are rejected
    because these surface checks cannot establish their intended nesting order.
    Unsupported model responses cannot execute and need no action admission.
    """
    import re

    if request["kind"] != "instruction":
        return {"status": "admitted", "checks": ["no executable action"], "reasons": []}
    text = instruction.lower().replace("’", "'")
    reasons = []
    if re.search(
        r"\b(?:not|never|avoid|without|unless|if|when|until)\b|\b(?:don't|cannot|can't)\b",
        text,
    ):
        reasons.append(
            "Negation, exclusion, or condition cannot be certified by this action grammar"
        )
    if re.search(
        r"\b(?:seconds?|minutes?|hours?|speed|deadline|quickly|slowly)\b", text
    ):
        reasons.append(
            "Timing or speed constraints cannot be represented by this action grammar"
        )
    concurrent = bool(
        re.search(
            r"\bwhile\b|\bsimultaneous(?:ly)?\b|\bconcurrently\b|at the same time|in parallel",
            text,
        )
    )
    sequential = bool(
        re.search(
            r"\b(?:then|before|after|afterwards|subsequently)\b|one at a time", text
        )
    )
    operation = request["tree"].operation
    if concurrent and sequential:
        reasons.append(
            "Mixed sequencing/concurrency requires a richer semantic verifier"
        )
    elif concurrent and operation != "parallel":
        reasons.append("Explicit concurrency cue disagrees with model instruction tree")
    elif sequential and operation != "sequence":
        reasons.append("Explicit sequencing cue disagrees with model instruction tree")
    return {
        "status": "rejected" if reasons else "admitted",
        "reasons": reasons,
        "checks": [
            "authored negation/condition cues",
            "authored sequence/concurrency cues",
        ],
        "limitation": "Passing these bounded cue checks does not prove semantic fidelity",
    }


def main():
    """Persist a model parse receipt for later sensor-grounded mission admission."""
    import argparse
    from dataclasses import asdict
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--prompt-version", choices=("v1", "v2"), default="v2")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing to overwrite an existing model parse receipt")
    prompt = SYSTEM_PROMPT if args.prompt_version == "v1" else V2_SYSTEM_PROMPT
    proposal = FrozenLanguageModel().propose(args.instruction, prompt)
    receipt = {
        "instruction": args.instruction,
        "prompt_version": args.prompt_version,
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "system_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "proposal": asdict(proposal),
        "meaning_review": "required; valid JSON does not establish semantic fidelity",
        "grounding": "not yet performed; executor must use fresh observed scene",
    }
    try:
        request = decode_request(proposal.raw_text)
        receipt["schema_valid"] = True
        receipt["semantic_admission"] = check_semantic_admission(
            args.instruction, request
        )
        receipt["request"] = {
            key: asdict(value) if key in {"tree", "reference"} else value
            for key, value in request.items()
        }
    except (ValueError, TypeError, KeyError) as exc:
        receipt["schema_valid"] = False
        receipt["error"] = str(exc)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps(
            {"output": str(args.output), "schema_valid": receipt["schema_valid"]}
        )
    )


if __name__ == "__main__":
    main()
