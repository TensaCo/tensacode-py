from dataclasses import dataclass
from collections.abc import Sequence
from types import MappingProxyType
from typing import Any, Mapping

from ._structured import InvalidModelOutput, StructuredOperation, finite_scores, model_configuration, optional_bool


@dataclass(frozen=True)
class RetrievalResult:
    """Selected item ``keys`` and ``items`` in rank order, with optional non-probability scores."""
    keys: tuple[str, ...]
    items: tuple[Any, ...]
    scores: Mapping[str, float] | None = None
    abstained: bool = False

    def __post_init__(self):
        object.__setattr__(self, "keys", tuple(self.keys))
        object.__setattr__(self, "items", tuple(self.items))
        if self.scores is not None and not isinstance(self.scores, MappingProxyType):
            object.__setattr__(self, "scores", MappingProxyType(dict(self.scores)))

    @property
    def distribution(self):
        """Retrieval scores have no probability interpretation."""
        return None


class Retrieve(StructuredOperation):
    """Select authored item keys using an owned seq2seq model.

    ``decoding='likelihood'`` scores every item description and returns the
    ``limit`` highest; its scores are log-likelihoods, not probabilities.
    ``from_model`` explicitly wraps an external provider without owned artifacts.
    """
    schema_name = "tensorcode.retrieve"

    def _alternatives(self):
        return [(f"{key}: {self.descriptions[key]}", self.descriptions[key]) for key in self.items]

    def _from_scores(self, scores):
        keyed = dict(zip(self.items, scores))
        keys = tuple(sorted(keyed, key=keyed.__getitem__, reverse=True)[: self.limit])
        return RetrievalResult(keys=keys, items=tuple(self.items[key] for key in keys), scores=keyed)

    def _target_weights(self, result):
        if result.abstained:
            raise ValueError("likelihood decoding has no abstention alternative")
        return [1.0 / len(result.keys) if key in result.keys else 0.0 for key in self.items]

    semantic_fields = {'instructions', 'limit', 'descriptions', 'items'}

    def _configure_semantics(self, config):
        super()._configure_semantics(config)
        items = config.get("items")
        descriptions = config.get("descriptions")
        limit = config.get("limit", 1)
        if not isinstance(items, Mapping) or not items:
            raise ValueError("items must be a nonempty mapping of stable string keys")
        if not all(isinstance(key, str) for key in items):
            raise TypeError("item keys must be strings")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= len(items):
            raise ValueError("limit must be between 1 and the item count")
        self.items = MappingProxyType(dict(items))
        if descriptions is None:
            descriptions = {
                key: item for key, item in self.items.items() if isinstance(item, str)
            }
        if (
            not isinstance(descriptions, Mapping)
            or set(descriptions) != set(self.items)
            or not all(isinstance(description, str) and description for description in descriptions.values())
        ):
            raise ValueError(
                "descriptions must provide nonempty text for every item; arbitrary item values are not stringified"
            )
        self.descriptions = MappingProxyType(dict(descriptions))
        self.limit = limit

    def response_schema(self):
        """JSON schema the model's structured output must satisfy."""
        keys = list(self.items)
        return {
            "type": "object",
            "properties": {
                "keys": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": keys,
                        "description": "Candidate meanings: "
                        + "; ".join(f"{key}: {self.descriptions[key]}" for key in keys),
                    },
                    "maxItems": self.limit,
                    "uniqueItems": True,
                },
                "scores": {
                    "type": ["object", "null"],
                    "properties": {key: {"type": "number"} for key in keys},
                    "required": keys,
                    "additionalProperties": False,
                },
                "abstained": {"type": "boolean"},
            },
            "required": ["keys", "scores", "abstained"],
            "additionalProperties": False,
        }

    def _parse(self, value):
        abstained = optional_bool(value, "abstained")
        keys = value.get("keys")
        if not isinstance(keys, Sequence) or isinstance(keys, (str, bytes)):
            raise InvalidModelOutput("keys must be a sequence")
        keys = tuple(keys)
        if not all(isinstance(key, str) for key in keys):
            raise InvalidModelOutput("Retrieved keys must be strings")
        if len(keys) > self.limit or len(set(keys)) != len(keys) or not set(keys) <= set(self.items):
            raise InvalidModelOutput("Retrieved keys must be unique configured items within the limit")
        if abstained and keys:
            raise InvalidModelOutput("An abstained retrieval must return no keys")
        if not abstained and not keys:
            raise InvalidModelOutput("A non-abstained retrieval must return at least one key")
        return RetrievalResult(
            keys=keys,
            items=tuple(self.items[key] for key in keys),
            scores=finite_scores(value.get("scores"), self.items),
            abstained=abstained,
        )

    def configuration(self):
        """JSON configuration that reconstructs this operation."""
        if self._owned:
            return super().configuration()
        return {
            "type": "text_retrieve",
            "item_keys": list(self.items),
            "descriptions": dict(self.descriptions),
            "limit": self.limit,
            "instructions": self.instructions,
            "model": model_configuration(self.model),
        }
