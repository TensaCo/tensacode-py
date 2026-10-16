import json

import pytest

from examples.robotics.warehouse.language_repair import propose_schema_valid
from examples.robotics.warehouse.model_language import ModelProposal


class FakeModel:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def propose(self, text, prompt):
        self.calls.append((text, prompt))
        return ModelProposal(next(self.responses), 1, 0.0, "unit-fixture")


def test_schema_repair_is_actual_second_proposal_without_expected_meaning():
    model = FakeModel(
        ['{"instruction":', '{"instruction":["move","parcel_0","receiver_0"]}']
    )
    result = propose_schema_valid(model, "Move parcel_0 to receiver_0", "schema prompt")
    assert result["schema_valid"]
    assert len(result["attempts"]) == 2
    assert result["attempts"][0]["proposal"]["raw_text"] == '{"instruction":'
    repair = json.loads(model.calls[1][0])
    assert set(repair) == {"original_request", "rejected_candidate", "schema_error"}
    assert repair["original_request"] == "Move parcel_0 to receiver_0"


def test_valid_but_wrong_candidate_is_never_repaired_using_gold_meaning():
    model = FakeModel(['{"instruction":["move","parcel_999","receiver_999"]}'])
    result = propose_schema_valid(model, "Move parcel_0 to receiver_0", "schema prompt")
    assert result["schema_valid"]
    assert result["request"]["tree"].object.name == "parcel_999"
    assert len(model.calls) == 1


def test_failed_repairs_are_retained_and_bounded():
    model = FakeModel(["bad first output", "bad second output"])
    result = propose_schema_valid(model, "original", "schema prompt")
    assert not result["schema_valid"]
    assert len(result["attempts"]) == 2
    with pytest.raises(ValueError):
        propose_schema_valid(model, "original", "prompt", max_attempts=3)
