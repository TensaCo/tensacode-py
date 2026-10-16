"""Prepare model candidates for scheduler checks within a controlled grammar.

This does not execute actions or establish arbitrary English equivalence. The
complete authored grammar must independently agree with the actual model output;
then references must resolve uniquely to fresh observed scene evidence. Scheduler
capability, resource, and motion-feasibility checks remain mandatory at dispatch.
"""

from .language_review import certify_request
from .model_language import decode_request, ground_request


def prepare_candidate(text, raw_candidate, scene, now, max_age=2.0):
    """Return a reviewable proposal, never replace a model candidate with grammar."""
    result = {"dispatch_authorized": False, "raw_candidate": raw_candidate}
    try:
        request = decode_request(raw_candidate)
    except (ValueError, TypeError, KeyError) as exc:
        return {**result, "status": "invalid", "reason": str(exc)}
    certificate = certify_request(text, request)
    result["semantic_certificate"] = certificate
    if certificate["status"] != "certified":
        return {**result, "status": "needs_review"}
    grounded = ground_request(request, scene, now, max_age)
    result["grounding"] = grounded
    if grounded["status"] != "supported":
        return {**result, "status": grounded["status"]}
    if request["kind"] == "query":
        return {**result, "status": "answered"}
    return {
        **result,
        "status": "ready_for_scheduler_checks",
        "instruction": request["tree"].to_dict(),
        "required_dispatch_checks": [
            "fresh_regrounding",
            "dependencies",
            "robot_capabilities",
            "exclusive_resources",
            "motion_feasibility",
        ],
    }
