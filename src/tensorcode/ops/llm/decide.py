from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from ._structured import InvalidModelOutput, StructuredOperation, model_configuration, optional_bool, optional_confidence, probability_distribution
from .classify import _selection_schema


@dataclass(frozen=True)
class DecisionResult:
    choice: str | None
    distribution: Mapping[str, float] | None = None
    confidence: float | None = None
    abstained: bool = False

    def __post_init__(self):
        if self.distribution is not None and not isinstance(self.distribution, MappingProxyType):
            object.__setattr__(self, "distribution", MappingProxyType(dict(self.distribution)))

    @property
    def value(self):
        return self.choice


class Decide(StructuredOperation):
    schema_name = "tensorcode.decide"

    def __init__(self, model, *, options, instructions=None):
        super().__init__(model, instructions=instructions)
        self.options = tuple(options)
        if not self.options or not all(isinstance(option, str) for option in self.options):
            raise ValueError("options must be nonempty strings")
        if len(set(self.options)) != len(self.options):
            raise ValueError("options must be unique")

    def response_schema(self):
        return _selection_schema("choice", self.options)

    def _parse(self, value):
        abstained = optional_bool(value, "abstained")
        choice = value.get("choice")
        if abstained:
            if choice is not None:
                raise InvalidModelOutput("An abstained decision must have choice null")
        elif choice not in self.options:
            raise InvalidModelOutput("Decision choice is not one of the configured options")
        return DecisionResult(
            choice=choice,
            distribution=probability_distribution(value.get("distribution"), self.options),
            confidence=optional_confidence(value),
            abstained=abstained,
        )

    def configuration(self):
        return {
            "type": "llm_decide",
            "options": list(self.options),
            "instructions": self.instructions,
            "model": model_configuration(self.model),
        }
