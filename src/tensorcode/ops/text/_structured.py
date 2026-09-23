from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Mapping
from math import isclose, isfinite
from types import MappingProxyType

from ..._internal.text.owned import OwnedTextOperation
from .messages import Message
from .model import ModelOutput, ModelRequest


class InvalidModelOutput(ValueError):
    """A model returned a value that does not satisfy an operation contract."""


def model_configuration(model):
    configure = getattr(model, "configuration", None)
    if callable(configure):
        configured = configure()
    else:
        model_type = type(model)
        configured = {"type": f"{model_type.__module__}.{model_type.__qualname__}"}
    try:
        json.dumps(configured)
    except (TypeError, ValueError) as exc:
        raise TypeError("model.configuration() must return JSON-safe data") from exc
    return configured


def message_sequence(value, context=None):
    try:
        messages = tuple(value)
    except TypeError as exc:
        raise TypeError("Expected a Message sequence") from exc
    if not messages or not all(isinstance(message, Message) for message in messages):
        raise TypeError("Expected a nonempty Message sequence")
    conditioning = []
    for group in (context or {}).values():
        try:
            group_messages = tuple(group)
        except TypeError as exc:
            raise TypeError("Context must contain Message sequences") from exc
        if not all(isinstance(message, Message) for message in group_messages):
            raise TypeError("Context must contain Message sequences")
        conditioning.extend(group_messages)
    return tuple(conditioning) + messages


def call_model(model, request):
    complete = getattr(model, "complete", None)
    if not callable(complete):
        raise TypeError("Structured operations require a model.complete(ModelRequest) method")
    output = complete(request)
    if not isinstance(output, ModelOutput):
        raise TypeError("model.complete must return ModelOutput")
    return output


def require_structured(output):
    if output.structured is None:
        raise InvalidModelOutput("Model did not supply structured output")
    return output.structured


def optional_bool(value, key):
    if key not in value:
        raise InvalidModelOutput(f"{key} must be supplied explicitly")
    raw = value[key]
    if not isinstance(raw, bool):
        raise InvalidModelOutput(f"{key} must be a boolean")
    return raw


def optional_confidence(value):
    if "confidence" not in value or value["confidence"] is None:
        return None
    confidence = value["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise InvalidModelOutput("confidence must be a number from 0 to 1")
    confidence = float(confidence)
    if not isfinite(confidence) or not 0 <= confidence <= 1:
        raise InvalidModelOutput("confidence must be a number from 0 to 1")
    return confidence


def probability_distribution(raw, expected_keys, *, key_transform=lambda key: key):
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise InvalidModelOutput("distribution must be a mapping")
    try:
        distribution = {key_transform(key): value for key, value in raw.items()}
    except (TypeError, ValueError) as exc:
        raise InvalidModelOutput("distribution contains invalid keys") from exc
    if set(distribution) != set(expected_keys):
        raise InvalidModelOutput("distribution keys must match the configured alternatives")
    normalized = {}
    for key, probability in distribution.items():
        if isinstance(probability, bool) or not isinstance(probability, (int, float)):
            raise InvalidModelOutput("distribution probabilities must be numbers")
        probability = float(probability)
        if not isfinite(probability) or not 0 <= probability <= 1:
            raise InvalidModelOutput("distribution probabilities must be between 0 and 1")
        normalized[key] = probability
    if not isclose(sum(normalized.values()), 1.0, abs_tol=1e-3):
        raise InvalidModelOutput("distribution probabilities must sum to 1")
    return MappingProxyType(normalized)


def finite_scores(raw, expected_keys):
    if raw is None:
        return None
    if not isinstance(raw, Mapping) or set(raw) != set(expected_keys):
        raise InvalidModelOutput("scores must contain exactly the configured item keys")
    result = {}
    for key, score in raw.items():
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise InvalidModelOutput("scores must be finite numbers")
        score = float(score)
        if not isfinite(score):
            raise InvalidModelOutput("scores must be finite numbers")
        result[key] = score
    return MappingProxyType(result)


def softmax(scores):
    top = max(scores)
    weights = [math.exp(score - top) for score in scores]
    total = sum(weights)
    return [weight / total for weight in weights]


def alternative_descriptions(raw, alternatives, name):
    """Validate optional nonempty descriptions keyed by configured alternatives."""
    if raw is None:
        return MappingProxyType({})
    if not isinstance(raw, Mapping) or not set(raw) <= set(alternatives):
        raise ValueError(f"descriptions must map configured {name} to text")
    if not all(isinstance(text, str) and text for text in raw.values()):
        raise ValueError("descriptions must be nonempty strings")
    return MappingProxyType(dict(raw))


class SelectionOperation:
    """Shared likelihood behavior for Classify and Decide alternatives."""

    def _alternatives(self):
        return [
            (f"{alternative}: {self.descriptions[alternative]}" if alternative in self.descriptions else alternative,
             alternative)
            for alternative in self._choices()
        ]

    def _from_scores(self, scores):
        choices = self._choices()
        probabilities = softmax(scores)
        best = max(range(len(choices)), key=probabilities.__getitem__)
        return self._result(
            choices[best],
            distribution=dict(zip(choices, probabilities)),
            confidence=probabilities[best],
            abstained=False,
        )

    def _target_weights(self, result):
        if result.abstained:
            raise ValueError("likelihood decoding has no abstention alternative")
        choices = self._choices()
        if result.distribution is not None:
            return [result.distribution[choice] for choice in choices]
        return [1.0 if choice == result.value else 0.0 for choice in choices]


class StructuredOperation(OwnedTextOperation):
    decoding_fields = frozenset({"decoding", "likelihood_normalization"})

    def _configure_decoding(self, config):
        self.decoding = config.get("decoding", "generate")
        if self.decoding not in ("generate", "likelihood"):
            raise ValueError("decoding must be 'generate' or 'likelihood'")
        self.likelihood_normalization = config.get("likelihood_normalization", "sum")
        if self.likelihood_normalization not in ("sum", "mean"):
            raise ValueError("likelihood_normalization must be 'sum' or 'mean'")
        if "likelihood_normalization" in config and self.decoding != "likelihood":
            raise ValueError("likelihood_normalization requires decoding='likelihood'")

    def _alternatives(self):
        """Return ``(display, target)`` text for each configured alternative."""
        raise NotImplementedError

    def _scoring_request(self, value, context):
        return ModelRequest(message_sequence(value, context), instructions=self.instructions)

    def _likelihood_forward(self, value, context):
        scores = self.model.score_alternatives(
            self._scoring_request(value, context),
            self._alternatives(),
            normalization=self.likelihood_normalization,
        )
        return self._from_scores(scores)

    def _likelihood_loss(self, value, targets, context):
        import torch

        if not isinstance(targets, Mapping):
            raise ValueError("structured targets must be an explicit JSON mapping")
        weights = torch.as_tensor(self._target_weights(self._parse(targets)), dtype=torch.float32)
        scores = self.model.alternative_log_likelihoods(
            self._scoring_request(value, context),
            self._alternatives(),
            normalization=self.likelihood_normalization,
        )
        weights = weights.to(device=scores.device, dtype=scores.dtype)
        return -(weights * scores.log_softmax(-1)).sum()

    def _target_weights(self, result):
        """Map a parsed target result to a probability vector over alternatives."""
        raise NotImplementedError

    def _request(self, value, context):
        return ModelRequest(
            message_sequence(value, context),
            instructions=self.instructions,
            response_schema=self.response_schema(),
            schema_name=self.schema_name,
        )

    def forward(self, value, *, context=None):
        if self.decoding == "likelihood":
            return self._likelihood_forward(value, context)
        return self._parse(require_structured(call_model(self.model, self._request(value, context))))

    async def aforward(self, value, *, context=None):
        if self.decoding == "likelihood":
            return await super(StructuredOperation, self).aforward(value, context=context)
        request = self._request(value, context)
        acomplete = getattr(self.model, "acomplete", None)
        if callable(acomplete):
            output = await acomplete(request)
            if not isinstance(output, ModelOutput):
                raise TypeError("model.acomplete must return ModelOutput")
        else:
            output = await super().aforward(value, context=context)
            return output
        return self._parse(require_structured(output))

    def batch(self, values, *, contexts=None):
        from tensorcode._internal.tracing import _active, invoke

        values = tuple(values)
        if contexts is None:
            contexts = (None,) * len(values)
        else:
            contexts = tuple(contexts)
            if len(contexts) != len(values):
                raise ValueError("contexts must match the number of values")
        # A fused provider call cannot reserve and complete ordinary trace calls
        # atomically. Preserve OutputRef unwrapping and failed-call capture by
        # using the normal per-item boundary whenever a Session is active.
        if _active.get() is not None:
            return tuple(self(value, context=context) for value, context in zip(values, contexts))
        complete_batch = getattr(self.model, "complete_batch", None)
        if not callable(complete_batch):
            return tuple(self(value, context=context) for value, context in zip(values, contexts))
        requests = tuple(self._request(value, context) for value, context in zip(values, contexts))
        outputs = tuple(complete_batch(requests))
        if len(outputs) != len(requests):
            raise InvalidModelOutput("Model batch result count does not match request count")
        if not all(isinstance(output, ModelOutput) for output in outputs):
            raise TypeError("model.complete_batch must return ModelOutput values")
        parsed = tuple(self._parse(require_structured(output)) for output in outputs)
        # Each result remains an ordinary operation boundary for tracing even
        # when the backend fused the transport call.
        return tuple(
            invoke(self, value, context, lambda _value, *, context=None, result=result: result)
            for value, context, result in zip(values, contexts, parsed)
        )

    async def abatch(self, values, *, contexts=None):
        values = tuple(values)
        if contexts is None:
            contexts = (None,) * len(values)
        else:
            contexts = tuple(contexts)
            if len(contexts) != len(values):
                raise ValueError("contexts must match the number of values")
        return tuple(
            await asyncio.gather(
                *(self.acall(value, context=context) for value, context in zip(values, contexts))
            )
        )
