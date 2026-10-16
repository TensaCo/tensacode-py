"""Full-language certificates and independent neural review for model proposals.

A model still produces every proposed instruction. The authored grammar is ONLY
an independent verifier: it never replaces a malformed or incorrect candidate.
Neural agreement is reported separately and is not an execution certificate.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict

from .language import Instruction
from .model_language import (
    V2_SYSTEM_PROMPT,
    _concise_reference,
    decode_request,
    strict_json,
)

V3_PROMPT = (
    V2_SYSTEM_PROMPT
    + """
Before translating, inspect the ENTIRE request for an unsupported constraint.
An exception (unless), permission condition (only if), future start time, deadline,
repeat count, prohibition, or requested speed means the whole request is unsupported.
Do not extract only its executable-looking fragment.
Once the first task is done / upon completion / only after completing means sequence.
Examples:
User: Deliver parcel_0 to receiver_0, and once that is done, deliver parcel_2 to receiver_1.
{"instruction":["sequence",["move","parcel_0","receiver_0"],["move","parcel_2","receiver_1"]]}
User: Move parcel_0 to receiver_0 unless the destination is occupied.
{"unsupported":"Conditional request cannot be represented"}
User: Put parcel_0 on receiver_0 in less than two seconds.
{"unsupported":"Deadline cannot be represented"}
User: In ten minutes, deliver parcel_0 to receiver_0.
{"unsupported":"Delayed jobs cannot be represented"}
User: Repeat the transfer twice.
{"unsupported":"Repetition counts and missing references are unsupported"}
"""
)

REVIEW_PROMPT = """Independently review a proposed warehouse interpretation.
The user request and a precise candidate meaning are provided. Do NOT repair the
candidate. Return ONLY {"equivalent":true|false,"reason":"short reason"}.
True requires every action, object, destination, order, concurrency requirement,
question, and restriction to be preserved. Missing a clause or inventing one is false.
'After', 'once done', 'upon completion' require order; parallel actions have no order.
A candidate move cannot preserve a conditional, deadline, delay, prohibition, speed,
or repeat count because this execution language has none of those operations.
A location query is not a mass/count question. Unknown references must stay unknown.
An unsupported response is equivalent only if the request needs unsupported abilities
or lacks the references needed to represent it; rejecting a clear supported move is false.
Do not assume a plausible executable plan is equivalent to the request.
"""


class UncertifiedLanguage(ValueError):
    """The entire utterance is outside the authored certificate grammar."""


def _clean(text):
    if not isinstance(text, str) or not 0 < len(text) <= 1500:
        raise UncertifiedLanguage("Request length is outside grammar bounds")
    text = " ".join(text.lower().strip().rstrip(".?!").split())
    return re.sub(r"^(?:please|could you|can you|would you)\s+", "", text, count=1)


def _compose(operation, pieces, depth):
    if not 2 <= len(pieces) <= 8:
        raise UncertifiedLanguage("Composition arity is outside grammar bounds")
    return Instruction(
        operation, children=tuple(_command(p, depth + 1) for p in pieces)
    )


def _command(text, depth=0):
    if depth > 8:
        raise UncertifiedLanguage("Composition depth exceeds grammar bounds")
    text = _clean(text)
    # Entire prefix temporal constructions; both sides must independently parse.
    match = re.fullmatch(r"(before|after) (.+), (.+)", text)
    if match:
        first, second = match[2], match[3]
        return _compose(
            "sequence",
            [second, first] if match[1] == "before" else [first, second],
            depth,
        )
    text = re.sub(r"^first\s+", "", text, count=1)
    sequence = re.split(
        r",?\s+(?:and\s+)?(?:once that is done|upon completion),?\s+|\s*;\s*(?:then|afterwards)?\s*|,?\s+(?:and\s+)?(?:then|afterwards)\s+",
        text,
    )
    if len(sequence) > 1:
        return _compose("sequence", sequence, depth)
    parallel = re.split(
        r",?\s+(?:while|concurrently with|simultaneously with|and at the same time)\s+",
        text,
    )
    if len(parallel) > 1:
        return _compose("parallel", parallel, depth)
    match = re.fullmatch(
        r"(?:move|moving|bring|bringing|carry|carrying|deliver|delivering|take|taking|put|putting|relocate|relocating|send|sending)\s+(.+)",
        text,
    )
    if not match:
        raise UncertifiedLanguage("No complete supported command production")
    candidates = []
    for separator in re.finditer(r"\s+(?:to|onto|on|at)\s+", match[1]):
        try:
            obj = _concise_reference(match[1][: separator.start()])
            dest = _concise_reference(match[1][separator.end() :])
            candidates.append(Instruction("pick_place", obj, dest))
        except ValueError:
            continue
    if len(candidates) != 1:
        raise UncertifiedLanguage(
            "Command does not have exactly one complete reference parse"
        )
    return candidates[0]


def certificate_parse(text):
    """Parse all tokens; unsupported clauses cannot disappear into a keyword match."""
    text = _clean(text)
    for pattern in (
        r"where is (.+)",
        r"where did you last observe (.+)",
        r"what is the current observed position of (.+)",
        r"tell me where (.+) is",
        r"tell me the location of (.+)",
    ):
        match = re.fullmatch(pattern, text)
        if match:
            try:
                return {"kind": "query", "reference": _concise_reference(match[1])}
            except ValueError as exc:
                raise UncertifiedLanguage(
                    "Question reference is outside grammar"
                ) from exc
    return {"kind": "instruction", "tree": _command(text)}


def signature(request):
    """Normalize associative composition and unordered parallel children."""

    def node(tree):
        if tree.operation == "pick_place":
            return ["move", asdict(tree.object), asdict(tree.destination)]
        children = []
        for child in tree.children:
            value = node(child)
            children.extend(value[1:] if value[0] == tree.operation else [value])
        if tree.operation == "parallel":
            children.sort(key=lambda x: json.dumps(x, sort_keys=True))
        return [tree.operation, *children]

    if request["kind"] == "instruction":
        return {"instruction": node(request["tree"])}
    if request["kind"] == "query":
        return {"where": asdict(request["reference"])}
    return {"unsupported": True}


def certify_request(text, request):
    try:
        expected = certificate_parse(text)
    except UncertifiedLanguage as exc:
        return {
            "status": "abstain",
            "reason": str(exc),
            "method": "complete_authored_grammar",
        }
    same = signature(expected) == signature(request)
    return {
        "status": "certified" if same else "rejected",
        "method": "complete_authored_grammar",
        "expected_signature": signature(expected),
        "candidate_signature": signature(request),
        "reason": "Complete grammar meaning matches model candidate"
        if same
        else "Model candidate contradicts complete grammar meaning",
    }


def review_input(text, request):
    return json.dumps(
        {"user_request": text, "candidate_meaning": signature(request)}, sort_keys=True
    )


def decode_review(raw):
    value = strict_json(raw)
    if (
        set(value) != {"equivalent", "reason"}
        or type(value["equivalent"]) is not bool
        or not isinstance(value["reason"], str)
        or not 0 < len(value["reason"]) <= 1000
    ):
        raise ValueError("Invalid independent-review schema")
    return value


def decode_candidate(raw):
    return decode_request(raw)
