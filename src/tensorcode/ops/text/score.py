from dataclasses import dataclass
from collections.abc import Mapping as MappingABC
from types import MappingProxyType
from typing import Mapping

from ._structured import InvalidModelOutput, StructuredOperation, model_configuration, optional_bool, optional_confidence, probability_distribution


@dataclass(frozen=True)
class ScoreResult:
    value: float | None
    distribution: Mapping[int, float] | None = None
    confidence: float | None = None
    abstained: bool = False

    def __post_init__(self):
        if self.distribution is not None and not isinstance(self.distribution, MappingProxyType):
            object.__setattr__(self, "distribution", MappingProxyType(dict(self.distribution)))


class Score(StructuredOperation):
    schema_name = "tensorcode.score"

    def __init__(self, model, *, rubric, instructions=None):
        super().__init__(model, instructions=instructions)
        self.rubric = tuple(rubric)
        if not self.rubric or not all(isinstance(level, str) for level in self.rubric):
            raise ValueError("rubric must contain one or more string levels")

    def response_schema(self):
        keys = tuple(str(index) for index in range(len(self.rubric)))
        return {
            "type": "object",
            "properties": {
                "score": {"type": ["number", "null"], "minimum": 0, "maximum": len(self.rubric) - 1},
                "distribution": {
                    "type": ["object", "null"],
                    "properties": {
                        key: {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                            "description": self.rubric[int(key)],
                        }
                        for key in keys
                    },
                    "required": list(keys),
                    "additionalProperties": False,
                },
                "confidence": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
                "abstained": {"type": "boolean"},
            },
            "required": ["score", "distribution", "confidence", "abstained"],
            "additionalProperties": False,
        }

    def _parse(self, value):
        abstained = optional_bool(value, "abstained")
        raw_score = value.get("score")
        if abstained:
            if raw_score is not None:
                raise InvalidModelOutput("An abstained score must have score null")
            score = None
        else:
            if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
                raise InvalidModelOutput("score must be a number")
            score = float(raw_score)
            if not 0 <= score <= len(self.rubric) - 1:
                raise InvalidModelOutput("score is outside the configured rubric")
        raw_distribution = value.get("distribution")
        if raw_distribution is not None and (
            not isinstance(raw_distribution, MappingABC)
            or set(raw_distribution) != {str(index) for index in range(len(self.rubric))}
        ):
            raise InvalidModelOutput("distribution keys must be canonical configured rubric indices")
        return ScoreResult(
            value=score,
            distribution=probability_distribution(
                raw_distribution, range(len(self.rubric)), key_transform=int
            ),
            confidence=optional_confidence(value),
            abstained=abstained,
        )

    def configuration(self):
        return {
            "type": "text_score",
            "rubric": list(self.rubric),
            "instructions": self.instructions,
            "model": model_configuration(self.model),
        }
