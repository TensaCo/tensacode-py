from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from ._structured import (
    InvalidModelOutput,
    StructuredOperation,
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


class Classify(StructuredOperation):
    schema_name = "tensorcode.classify"

    def __init__(self, model, *, labels, instructions=None):
        super().__init__(model, instructions=instructions)
        self.labels = tuple(labels)
        if not self.labels or not all(isinstance(label, str) for label in self.labels):
            raise ValueError("labels must be nonempty strings")
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("labels must be unique")

    def response_schema(self):
        return _selection_schema("label", self.labels)

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
        return {
            "type": "text_classify",
            "labels": list(self.labels),
            "instructions": self.instructions,
            "model": model_configuration(self.model),
        }


def _selection_schema(field, alternatives):
    distribution = {
        "type": ["object", "null"],
        "properties": {label: {"type": "number", "minimum": 0, "maximum": 1} for label in alternatives},
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
