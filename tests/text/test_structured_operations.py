import asyncio

import pytest

from tensorcode import trace
from tensorcode.ops import text as text_ops


class ScriptedModel:
    def __init__(self, *outputs):
        self.outputs = list(outputs)
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        return self.outputs.pop(0)


def messages(text="example"):
    return (text_ops.Message("user", text),)


def test_classify_returns_only_provider_supplied_distribution():
    model = ScriptedModel(
        text_ops.ModelOutput(
            structured={
                "label": "urgent",
                "distribution": {"routine": 0.1, "urgent": 0.9},
                "abstained": False,
            }
        ),
        text_ops.ModelOutput(structured={"label": "routine", "abstained": False}),
    )
    classify = text_ops.Classify.from_model(
        model, labels=("routine", "urgent"), instructions="Assess urgency"
    )

    supplied = classify(messages())
    missing = classify(messages("second"))

    assert supplied == text_ops.ClassificationResult(
        label="urgent", distribution={"routine": 0.1, "urgent": 0.9}
    )
    assert missing.label == "routine"
    assert missing.distribution is None
    assert missing.confidence is None
    assert model.requests[0].schema_name == "tensorcode.classify"


def test_classify_validates_labels_and_probability_distribution():
    classify = text_ops.Classify.from_model(
        ScriptedModel(
            text_ops.ModelOutput(
                structured={
                    "label": "invented",
                    "distribution": {"routine": 0.4, "urgent": 0.6},
                    "abstained": False,
                }
            )
        ),
        labels=("routine", "urgent"),
    )
    with pytest.raises(text_ops.InvalidModelOutput, match="configured labels"):
        classify(messages())

    classify = text_ops.Classify.from_model(
        ScriptedModel(
            text_ops.ModelOutput(
                structured={
                    "label": "urgent",
                    "distribution": {"routine": 0.4, "urgent": 0.4},
                    "abstained": False,
                }
            )
        ),
        labels=("routine", "urgent"),
    )
    with pytest.raises(text_ops.InvalidModelOutput, match="sum to 1"):
        classify(messages())


def test_classify_represents_explicit_abstention_without_distribution():
    result = text_ops.Classify.from_model(
        ScriptedModel(text_ops.ModelOutput(structured={"label": None, "abstained": True})),
        labels=("yes", "no"),
    )(messages())
    assert result == text_ops.ClassificationResult(label=None, abstained=True)


def test_structured_result_requires_explicit_abstention_state():
    classify = text_ops.Classify.from_model(
        ScriptedModel(text_ops.ModelOutput(structured={"label": "yes"})),
        labels=("yes", "no"),
    )
    with pytest.raises(text_ops.InvalidModelOutput, match="abstained"):
        classify(messages())


def test_score_validates_rubric_distribution_and_preserves_provider_confidence():
    score = text_ops.Score.from_model(
        ScriptedModel(
            text_ops.ModelOutput(
                structured={
                    "score": 1.7,
                    "distribution": {"0": 0.1, "1": 0.1, "2": 0.8},
                    "confidence": 0.81,
                    "abstained": False,
                }
            )
        ),
        rubric=("can wait", "this week", "today"),
        instructions="Assess urgency",
    )
    result = score(messages())
    assert result == text_ops.ScoreResult(
        value=1.7,
        distribution={0: 0.1, 1: 0.1, 2: 0.8},
        confidence=0.81,
    )


def test_score_rejects_noncanonical_or_colliding_distribution_keys():
    score = text_ops.Score.from_model(
        ScriptedModel(
            text_ops.ModelOutput(
                structured={
                    "score": 1.0,
                    "distribution": {"0": 0.25, "00": 0.25, "1": 0.5},
                    "abstained": False,
                }
            )
        ),
        rubric=("low", "high"),
    )
    with pytest.raises(text_ops.InvalidModelOutput, match="keys"):
        score(messages())


def test_decide_rejects_unconfigured_choice_and_allows_abstention():
    decide = text_ops.Decide.from_model(
        ScriptedModel(
            text_ops.ModelOutput(
                structured={"choice": "delete", "abstained": False}
            )
        ),
        options=("archive", "reply"),
    )
    with pytest.raises(text_ops.InvalidModelOutput, match="configured options"):
        decide(messages())

    abstained = text_ops.Decide.from_model(
        ScriptedModel(
            text_ops.ModelOutput(structured={"choice": None, "abstained": True})
        ),
        options=("archive", "reply"),
    )(messages())
    assert abstained == text_ops.DecisionResult(choice=None, abstained=True)


def test_retrieve_returns_only_configured_items_and_keeps_scores_semantically_distinct():
    retrieve = text_ops.Retrieve.from_model(
        ScriptedModel(
            text_ops.ModelOutput(
                structured={
                    "keys": ["policy"],
                    "scores": {"policy": 2.4, "faq": -1.0},
                    "abstained": False,
                }
            )
        ),
        items={"policy": {"text": "refund policy"}, "faq": "general"},
        descriptions={"policy": "refund policy", "faq": "general questions"},
        limit=1,
    )
    result = retrieve(messages("refund"))
    assert result.keys == ("policy",)
    assert result.items == ({"text": "refund policy"},)
    assert result.scores == {"policy": 2.4, "faq": -1.0}
    assert result.distribution is None


def test_retrieve_reports_non_string_model_keys_as_invalid_output():
    retrieve = text_ops.Retrieve.from_model(
        ScriptedModel(
            text_ops.ModelOutput(
                structured={"keys": [{}], "scores": None, "abstained": False}
            )
        ),
        items={"a": "first"},
    )
    with pytest.raises(text_ops.InvalidModelOutput, match="keys"):
        retrieve(messages())


def test_structured_operation_rejects_text_only_output_instead_of_guessing_json():
    classify = text_ops.Classify.from_model(
        ScriptedModel(text_ops.ModelOutput(text='{"label": "yes"}')),
        labels=("yes", "no"),
    )
    with pytest.raises(text_ops.InvalidModelOutput, match="structured"):
        classify(messages())


def test_acall_is_explicit_and_sync_call_never_returns_awaitable():
    classify = text_ops.Classify.from_model(
        ScriptedModel(
            text_ops.ModelOutput(structured={"label": "a", "abstained": False}),
            text_ops.ModelOutput(structured={"label": "b", "abstained": False}),
        ),
        labels=("a", "b"),
    )
    sync_result = classify(messages("sync"))
    async_result = asyncio.run(classify.acall(messages("async")))
    assert sync_result.label == "a"
    assert async_result.label == "b"


def test_batch_preserves_input_order_and_requires_exact_result_count():
    class BatchModel(ScriptedModel):
        def complete_batch(self, requests):
            self.requests.extend(requests)
            return (
                text_ops.ModelOutput(structured={"label": "a", "abstained": False}),
                text_ops.ModelOutput(structured={"label": "b", "abstained": False}),
            )

    classify = text_ops.Classify.from_model(BatchModel(), labels=("a", "b"))
    results = classify.batch((messages("one"), messages("two")))
    assert tuple(result.label for result in results) == ("a", "b")

    class BadBatchModel(BatchModel):
        def complete_batch(self, requests):
            return ()

    with pytest.raises(text_ops.InvalidModelOutput, match="batch result count"):
        text_ops.Classify.from_model(BadBatchModel(), labels=("a", "b")).batch(
            (messages("one"),)
        )


def test_batch_falls_back_to_per_item_calls_inside_a_trace():
    class TraceSafeModel(ScriptedModel):
        def complete_batch(self, requests):
            raise AssertionError("backend batch must be disabled while tracing")

    classify = text_ops.Classify.from_model(
        TraceSafeModel(
            text_ops.ModelOutput(structured={"label": "a", "abstained": False})
        ),
        labels=("a", "b"),
    )
    with trace() as session:
        encoded = text_ops.TextEncoder()("one")
        encoded_ref = session.calls[-1].output
        results = classify.batch((encoded_ref,))
    assert results[0].label == "a"
    assert len(session.calls) == 2


def test_batch_records_provider_failures_inside_a_trace():
    class FailingModel:
        def complete(self, request):
            raise RuntimeError("offline")

        def complete_batch(self, requests):
            raise AssertionError("backend batch must be disabled while tracing")

    classify = text_ops.Classify.from_model(FailingModel(), labels=("a", "b"))
    with trace() as session:
        with pytest.raises(RuntimeError, match="offline"):
            classify.batch((messages(),))
    assert len(session.calls) == 1
    assert session.calls[0].error == "RuntimeError: offline"


def test_structured_results_with_distributions_are_traceable():
    classify = text_ops.Classify.from_model(
        ScriptedModel(
            text_ops.ModelOutput(
                structured={
                    "label": "a",
                    "distribution": {"a": 0.75, "b": 0.25},
                    "abstained": False,
                }
            )
        ),
        labels=("a", "b"),
    )
    with trace() as session:
        result = classify(messages())
    assert result.distribution == {"a": 0.75, "b": 0.25}
    assert len(session.calls) == 1


def test_abatch_supports_async_only_models():
    class AsyncOnlyModel:
        async def acomplete(self, request):
            label = request.messages[-1].content
            return text_ops.ModelOutput(
                structured={"label": label, "abstained": False}
            )

    classify = text_ops.Classify.from_model(AsyncOnlyModel(), labels=("a", "b"))
    results = asyncio.run(classify.abatch((messages("a"), messages("b"))))
    assert tuple(result.label for result in results) == ("a", "b")
