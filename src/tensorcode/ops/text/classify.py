from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from ._structured import (
    InvalidModelOutput,
    SelectionOperation,
    StructuredOperation,
    alternative_descriptions,
    optional_bool,
    optional_confidence,
    model_configuration,
    probability_distribution,
)


@dataclass(frozen=True)
class ClassificationResult:
    label: str | None
    distribution: Mapping[str, float] | None = None
    confidence: float | None = None
    abstained: bool = False

    def __post_init__(self):
        if self.distribution is not None and not isinstance(self.distribution, MappingProxyType):
            object.__setattr__(self, "distribution", MappingProxyType(dict(self.distribution)))

    @property
    def value(self):
        return self.label


class Classify(SelectionOperation, StructuredOperation):
    """Classify messages with an owned seq2seq model and explicit labels.

    ``decoding='likelihood'`` scores every label in one encoder pass and always
    returns a full distribution; the default generates a JSON response.
    ``from_model`` explicitly wraps an external provider without owned artifacts.
    """
    schema_name = "tensorcode.classify"

    semantic_fields = {'labels', 'instructions', 'descriptions'}
    _result = ClassificationResult

    def _choices(self):
        return self.labels

    def _configure_semantics(self, config):
        super()._configure_semantics(config)
        labels = config.get("labels", [])
        if not isinstance(labels, (list, tuple)):
            raise ValueError("labels must be a sequence of strings")
        self.labels = tuple(labels)
        if not self.labels or not all(isinstance(label, str) and label for label in self.labels):
            raise ValueError("labels must be nonempty strings")
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("labels must be unique")
        self.descriptions = alternative_descriptions(config.get("descriptions"), self.labels, "labels")

    def response_schema(self):
        return _selection_schema("label", self.labels, self.descriptions)

    def _parse(self, value):
        abstained = optional_bool(value, "abstained")
        label = value.get("label")
        if abstained:
            if label is not None:
                raise InvalidModelOutput("An abstained classification must have label null")
        elif label not in self.labels:
            raise InvalidModelOutput("Classification label is not one of the configured labels")
        return ClassificationResult(
            label=label,
            distribution=probability_distribution(value.get("distribution"), self.labels),
            confidence=optional_confidence(value),
            abstained=abstained,
        )

    def configuration(self):
        if self._owned:
            return super().configuration()
        return {
            "type": "text_classify",
            "labels": list(self.labels),
            **({"descriptions": dict(self.descriptions)} if self.descriptions else {}),
            "instructions": self.instructions,
            "model": model_configuration(self.model),
        }


def _selection_schema(field, alternatives, descriptions=None):
    descriptions = descriptions or {}
    distribution = {
        "type": ["object", "null"],
        "properties": {
            label: {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                **({"description": descriptions[label]} if label in descriptions else {}),
            }
            for label in alternatives
        },
        "required": list(alternatives),
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            field: {"type": ["string", "null"], "enum": [*alternatives, None]},
            "distribution": distribution,
            "confidence": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
            "abstained": {"type": "boolean"},
        },
        "required": [field, "distribution", "confidence", "abstained"],
        "additionalProperties": False,
    }
