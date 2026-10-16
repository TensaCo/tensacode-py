"""Bounded model-only repair of invalid schemas, never semantic substitution."""

import json
from dataclasses import asdict

from .model_language import decode_request

REPAIR_INSTRUCTION = """
The next user message is a JSON repair request with original_request,
rejected_candidate, and schema_error. Translate original_request again using the
same supported schema. Repair the invalid format/arity while preserving the
ENTIRE original meaning. Emit one complete JSON object, no commentary. Do not
copy an invalid subtree merely to retain the previous layout. You are given no
correct answer. Unsupported original requests must remain unsupported.
"""


def propose_schema_valid(model, text, prompt, max_attempts=2, initial_proposal=None):
    """Return all raw attempts and schema status; semantic admission is separate.

    ``initial_proposal`` is for replaying a hash-bound retained model result in
    evaluation. Ordinary callers omit it and the model generates the first pass.
    Only decoder exceptions trigger another generation. A valid but incorrect
    interpretation is never silently retried using an authored expected meaning.
    """
    if type(max_attempts) is not int or not 1 <= max_attempts <= 2:
        raise ValueError("At most two actual model proposals are allowed")
    proposal = initial_proposal or model.propose(text, prompt)
    attempts = []
    for index in range(max_attempts):
        attempt = {"proposal": asdict(proposal)}
        attempts.append(attempt)
        try:
            request = decode_request(proposal.raw_text)
            attempt["schema_valid"] = True
            return {"schema_valid": True, "request": request, "attempts": attempts}
        except (ValueError, TypeError, KeyError) as exc:
            attempt.update(schema_valid=False, schema_error=str(exc))
        if index + 1 < max_attempts:
            repair_input = json.dumps(
                {
                    "original_request": text,
                    "rejected_candidate": proposal.raw_text,
                    "schema_error": attempt["schema_error"],
                }
            )
            proposal = model.propose(repair_input, prompt + REPAIR_INSTRUCTION)
    return {"schema_valid": False, "attempts": attempts}
