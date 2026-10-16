"""Model query intents; answers come only from observed scene or scheduler records."""

from __future__ import annotations

import re

from .model_language import decode_request, ground_request, strict_json

QUESTION_PROMPT = """Translate a warehouse question into exactly one JSON query.
Never answer the question and never invent a cause or a location.
Forms:
{"where":"object reference"} for observed locations.
{"why_waiting":"explicit_robot_id"} for why a robot is waiting, idle, or blocked.
{"unsupported":"brief reason"} for all other questions or missing references.
Preserve explicit IDs. 'robot r1' means r1. 'the first robot' has no explicit ID.
References support colors and left of/right of/in/on relations; box/package/carton
means parcel. A query about a robot does not establish that robot exists.
Do not turn an action request into a question. No battery, mass, price, future
prediction, counterfactual, task-count, or unobserved-cause questions are supported.
Examples:
User: What's holding up robot r1?
{"why_waiting":"r1"}
User: Why hasn't r2 been assigned work?
{"why_waiting":"r2"}
User: Where did you observe parcel_0?
{"where":"parcel_0"}
User: Is r1 out of battery?
{"unsupported":"Battery questions are not supported"}
"""


def decode_question(raw: str) -> dict:
    value = strict_json(raw)
    if set(value) == {"why_waiting"}:
        robot_id = value["why_waiting"]
        if not isinstance(robot_id, str) or not re.fullmatch(
            r"[a-z0-9_-]{1,64}", robot_id
        ):
            raise ValueError("Lifecycle query requires an explicit bounded robot ID")
        return {"kind": "why_waiting", "robot_id": robot_id}
    request = decode_request(raw)
    if request["kind"] not in {"query", "unsupported"}:
        raise ValueError("Action proposals are not accepted by the question interface")
    return request


def certify_question(text: str, request: dict) -> dict:
    """Compare an actual candidate with a complete controlled question grammar.

    Never remove extra clauses or repair model intent. Location questions use
    the existing full grammar; lifecycle questions require explicit robot IDs.
    Arbitrary English and unsupported-cause qualifiers require human review.
    """
    from .language_review import certify_request

    if request.get("kind") == "query":
        return certify_request(text, request)
    method = "complete_authored_question_grammar"
    if not isinstance(text, str) or not 0 < len(text) <= 1500:
        return {"status": "abstain", "method": method}
    normalized = " ".join(text.lower().strip().split())
    normalized = re.sub(r"[?.]$", "", normalized)
    robot = r"(?:robot )?([a-z][a-z0-9_-]{0,63})"
    patterns = (
        rf"why is {robot} (?:waiting|idle|blocked)",
        rf"what(?:'s| is) holding up {robot}",
        rf"why hasn(?:'t|’t) {robot} been assigned work",
        rf"what is preventing {robot} from getting started",
        rf"explain why {robot} cannot begin its transfer yet",
        rf"can you account for {robot} being blocked",
    )
    expected = None
    for pattern in patterns:
        match = re.fullmatch(pattern, normalized)
        if match:
            expected = {"kind": "why_waiting", "robot_id": match[1]}
            break
    if expected is None:
        return {
            "status": "abstain",
            "method": method,
            "reason": "Entire question is outside the controlled grammar",
        }
    return {
        "status": "certified" if request == expected else "rejected",
        "method": method,
        "expected_intent": expected,
        "candidate_intent": request,
    }


def answer_mission_question(request: dict, scene, now: float, scheduler=None) -> dict:
    """Read evidence only; never ask a model to supply the reason or answer text."""
    if request["kind"] == "why_waiting":
        if scheduler is None:
            return {"status": "unknown", "reason": "No scheduler evidence is available"}
        # explain consumes this controlled query, not original untrusted text.
        answer = scheduler.explain(f"Why is {request['robot_id']} waiting?")
        return {
            "query": {"kind": "why_waiting", "robot_id": request["robot_id"]},
            "answer_source": "scheduler_records",
            **answer,
        }
    return ground_request(request, scene, now)
