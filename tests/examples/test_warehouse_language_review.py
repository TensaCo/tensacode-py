"""Meaning certificates verify whole requests; they never replace model output."""

import json

import pytest

from examples.robotics.warehouse.language_review import (
    certify_request,
    decode_review,
    signature,
)
from examples.robotics.warehouse.model_language import decode_request

A = ["move", "parcel_0", "receiver_0"]
B = ["move", "parcel_1", "receiver_1"]
C = ["move", "parcel_2", "receiver_0"]


def request(tree):
    return decode_request(json.dumps({"instruction": tree}))


def test_unpunctuated_mixed_temporal_scope_abstains_without_choosing_precedence():
    text = "Move parcel_0 to receiver_0 while moving parcel_1 to receiver_1 then move parcel_2 to receiver_0"
    for tree in [
        ["sequence", ["parallel", A, B], C],
        ["parallel", A, ["sequence", B, C]],
    ]:
        assert certify_request(text, request(tree))["status"] == "abstain"
    explicit = text.replace(" then ", "; then ")
    assert (
        certify_request(explicit, request(["sequence", ["parallel", A, B], C]))[
            "status"
        ]
        == "certified"
    )


@pytest.mark.parametrize(
    "text",
    [
        "Move parcel_0 to receiver_0, and once that is done, move parcel_1 to receiver_1.",
        "Move parcel_0 to receiver_0; afterwards move parcel_1 to receiver_1.",
        "Before moving parcel_1 to receiver_1, take parcel_0 to receiver_0.",
        "After moving parcel_0 to receiver_0, take parcel_1 to receiver_1.",
    ],
)
def test_complete_temporal_parse_rejects_wrong_parallel_meaning(text):
    candidate = request(["parallel", A, B])
    original = signature(candidate)
    assert certify_request(text, candidate)["status"] == "rejected"
    assert signature(candidate) == original
    assert certify_request(text, request(["sequence", A, B]))["status"] == "certified"
    assert certify_request(text, request(["sequence", B, A]))["status"] == "rejected"


def test_nested_parallel_before_next_job_has_precise_dependencies():
    text = "Move parcel_0 to receiver_0 while moving parcel_1 to receiver_1; then take parcel_2 to receiver_0."
    assert (
        certify_request(text, request(["sequence", ["parallel", B, A], C]))["status"]
        == "certified"
    )
    assert certify_request(text, request(["parallel", A, B, C]))["status"] == "rejected"
    assert certify_request(text, request(["sequence", A, B, C]))["status"] == "rejected"


@pytest.mark.parametrize(
    "text",
    [
        "Move parcel_0 to receiver_0 unless the station is occupied.",
        "Move parcel_0 to receiver_0 provided that no robot is nearby.",
        "Move parcel_0 to receiver_0 in less than two seconds.",
        "After 30 seconds, move parcel_0 to receiver_0.",
        "Do not move parcel_0 to receiver_0.",
        "Repeat moving parcel_0 to receiver_0 three times.",
        "Move parcel_0 to receiver_0 and erase the task log.",
    ],
)
def test_complete_grammar_never_discards_unrepresented_clauses(text):
    assert certify_request(text, request(A))["status"] == "abstain"


def test_neural_review_only_accepts_exact_data_schema():
    assert (
        decode_review('{"equivalent":false,"reason":"Order changed"}')["equivalent"]
        is False
    )
    for value in [
        {"equivalent": "true", "reason": "Looks good"},
        {"equivalent": True, "reason": "Good", "execute": True},
        {"equivalent": True, "reason": ""},
    ]:
        with pytest.raises(ValueError):
            decode_review(json.dumps(value))


def test_semantic_signature_flattens_only_same_associative_operator():
    assert signature(request(["sequence", A, ["sequence", B, C]])) == signature(
        request(["sequence", A, B, C])
    )
    assert signature(request(["parallel", A, B])) == signature(
        request(["parallel", B, A])
    )
    assert signature(request(["sequence", A, B])) != signature(
        request(["sequence", B, A])
    )


def test_query_certificate_preserves_reference():
    correct = decode_request('{"where":"parcel left of parcel_1"}')
    wrong = decode_request('{"where":"parcel_1"}')
    text = "Tell me where the package left of parcel_1 is."
    assert certify_request(text, correct)["status"] == "certified"
    assert certify_request(text, wrong)["status"] == "rejected"


def test_paired_label_review_abstains_on_label_bias_and_low_mass():
    from examples.robotics.warehouse.language_pair_review import decide

    assert decide([0.95, 0.94], [0.8, 0.9]) == "accept"
    assert decide([0.04, 0.06], [0.8, 0.9]) == "reject"
    assert decide([0.95, 0.04], [0.8, 0.9]) == "abstain"
    assert decide([0.95, 0.94], [0.01, 0.9]) == "abstain"
    assert decide([0.81, 0.79], [0.8, 0.9]) == "abstain"
    with pytest.raises(ValueError, match="Both label mappings"):
        decide([0.9], [0.9])
