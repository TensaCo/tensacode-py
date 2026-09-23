from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from ._structured import InvalidModelOutput, SelectionOperation, StructuredOperation, alternative_descriptions, model_configuration, optional_bool, optional_confidence, probability_distribution
from .classify import _selection_schema


@dataclass(frozen=True)
class DecisionResult:
    """Selected ``choice`` (``None`` when abstained) with optional distribution and confidence."""
    choice: str | None
    distribution: Mapping[str, float] | None = None
    confidence: float | None = None
    abstained: bool = False

    def __post_init__(self):
        if self.distribution is not None and not isinstance(self.distribution, MappingProxyType):
            object.__setattr__(self, "distribution", MappingProxyType(dict(self.distribution)))

    @property
    def value(self):
        """The selected choice (alias used by generic choosers)."""
        return self.choice


class Decide(SelectionOperation, StructuredOperation):
    """Choose an authored option using an owned seq2seq model.

    ``decoding='likelihood'`` scores every option in one encoder pass; the
    default generates a JSON response. ``from_model`` explicitly wraps an
    external provider without owned artifacts.
    """
    schema_name = "tensorcode.decide"

    semantic_fields = {'instructions', 'options', 'descriptions'}
    _result = DecisionResult

    def _choices(self):
        return self.options

    def _configure_semantics(self, config):
        super()._configure_semantics(config)
        options = config.get("options", [])
        if not isinstance(options, (list, tuple)):
            raise ValueError("options must be a sequence of strings")
        self.options = tuple(options)
        if not self.options or not all(isinstance(option, str) and option for option in self.options):
            raise ValueError("options must be nonempty strings")
        if len(set(self.options)) != len(self.options):
            raise ValueError("options must be unique")
        self.descriptions = alternative_descriptions(config.get("descriptions"), self.options, "options")

    def response_schema(self):
        """JSON schema the model's structured output must satisfy."""
        return _selection_schema("choice", self.options, self.descriptions)

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
        """JSON configuration that reconstructs this operation."""
        if self._owned:
            return super().configuration()
        return {
            "type": "text_decide",
            "options": list(self.options),
            **({"descriptions": dict(self.descriptions)} if self.descriptions else {}),
            "instructions": self.instructions,
            "model": model_configuration(self.model),
        }
