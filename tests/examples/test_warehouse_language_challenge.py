import json

import pytest

from examples.robotics.warehouse import language_challenge as challenge
from examples.robotics.warehouse.model_language import ModelProposal


def test_challenge_retains_model_failure_and_refuses_overwrite(tmp_path, monkeypatch):
    class FakeModel:
        class torch:
            __version__ = "fixture"

        def __init__(self, device):
            assert device == "cpu"

        def propose(self, text, prompt):
            return ModelProposal("malformed", 1, 0.0, "fixture")

    monkeypatch.setattr(challenge, "TextCampaignModel", FakeModel)
    protocol = {
        "model": challenge.TEXT_MODEL_ID,
        "revision": challenge.TEXT_MODEL_REVISION,
        "prompt": "unit test",
        "cases": [
            {
                "id": "unit",
                "text": "Move parcel_0 to receiver_0",
                "expected": {"instruction": ["move", "parcel_0", "receiver_0"]},
            }
        ],
    }
    challenge.run(protocol, tmp_path, "cpu")
    rows = json.loads((tmp_path / "predictions.json").read_text())
    assert rows[0]["proposal"]["raw_text"] == "malformed"
    assert rows[0]["semantic_correct"] is False
    assert rows[0]["authored_grammar"]["semantic_correct"] is True
    assert "certificate" not in rows[0]
    with pytest.raises(ValueError, match="overwrite"):
        challenge.run(protocol, tmp_path, "cpu")


def test_challenge_rejects_unpinned_foundation_before_inference(tmp_path):
    with pytest.raises(ValueError, match="pinned"):
        challenge.run({"model": "unknown", "revision": "main"}, tmp_path, "cpu")
