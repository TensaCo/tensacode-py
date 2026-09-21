import pytest

from tensorcode.ops import llm
from tensorcode.runtime import DecisionPipeline


class Model:
    def __init__(self, structured):
        self.structured = structured
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        return llm.ModelOutput(structured=self.structured)


def test_ready_decision_configures_public_message_operations_and_keeps_distribution():
    model = Model(
        {
            "label": "billing",
            "distribution": {"billing": 0.75, "technical": 0.25},
            "abstained": False,
        }
    )
    tool = DecisionPipeline(
        model=model,
        labels=("billing", "technical"),
        instructions="Route this support request",
    )

    result = tool("I was charged twice")

    assert result.value == "billing"
    assert result.distribution == {"billing": 0.75, "technical": 0.25}
    assert isinstance(tool.encode, llm.TextEncoder)
    assert isinstance(tool.decide, llm.Classify)
    assert model.requests[0].instructions == "Route this support request"


def test_ready_decision_applies_replaceable_selection_policy():
    model = Model(
        {
            "label": "billing",
            "distribution": {"billing": 0.51, "technical": 0.49},
            "abstained": False,
        }
    )
    seen = []

    def require_margin(result):
        seen.append(result)
        return llm.ClassificationResult(
            label=None,
            distribution=result.distribution,
            abstained=True,
        )

    result = DecisionPipeline(
        model=model,
        labels=("billing", "technical"),
        selection_policy=require_margin,
    )("ambiguous")

    assert result.abstained
    assert result.value is None
    assert result.distribution == {"billing": 0.51, "technical": 0.49}
    assert seen[0].value == "billing"


def test_decision_retains_explicit_encode_decide_composition():
    calls = []

    def encode(value):
        calls.append(("encode", value))
        return value.upper()

    def decide(value, *, context=None):
        calls.append(("decide", value, context))
        return "chosen"

    result = DecisionPipeline(encode=encode, decide=decide)("input", context={"x": 1})

    assert result == "chosen"
    assert calls == [("encode", "input"), ("decide", "INPUT", {"x": 1})]


def test_decision_rejects_ambiguous_ready_and_explicit_configuration():
    with pytest.raises(ValueError, match="either"):
        DecisionPipeline(
            model=Model({}),
            labels=("a", "b"),
            encode=lambda value: value,
            decide=lambda value, context=None: value,
        )


def test_ready_decision_rejects_selection_policy_output_outside_labels():
    tool = DecisionPipeline(
        model=Model({"label": "a", "abstained": False}),
        labels=("a", "b"),
        selection_policy=lambda result: llm.ClassificationResult("invented"),
    )

    with pytest.raises(ValueError, match="configured labels"):
        tool("input")
