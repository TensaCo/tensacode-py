"""Learned token boundaries anchored to unchanged source characters.

Whitespace is an explicit representational gap: tokens cannot contain it and no
non-whitespace character may disappear. All other boundaries are proposed from
learned weights. No word, contraction, punctuation, or quotation rule is supplied.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import random
import tempfile
from typing import Any, Iterable, Mapping
import unicodedata

from .learned_parser import Perceptron

DEFAULT_MODEL_PATH = Path.home() / ".cache/tensorcode/models/ud_ewt_segmenter.json"
LABELS = ("start", "join")
ALGORITHM = "averaged-perceptron-character-boundaries"
REPRESENTATION = "unicode-codepoint-spans; whitespace-forced-gaps"


@dataclass(frozen=True)
class SegmentationCandidate:
    spans: tuple[tuple[int, int], ...]
    score: float
    provenance: tuple[str, ...] = ("learned:character-boundaries",)
    score_kind: str = "uncalibrated"


@dataclass(frozen=True)
class SegmentationSearch:
    candidates: tuple[SegmentationCandidate, ...]
    expansions: int
    truncated: bool
    complete: bool
    reason: str | None = None


def _validate_spans(text: str, spans: tuple[tuple[int, int], ...]) -> None:
    """Structural source fidelity, independent of any linguistic boundary choice."""
    if not isinstance(text, str):
        raise TypeError("segmentation source must be text")
    covered = bytearray(len(text))
    previous_end = 0
    for span in spans:
        if not isinstance(span, (tuple, list)) or len(span) != 2:
            raise ValueError("token spans must be start/end pairs")
        start, end = span
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text):
            raise ValueError("token span is outside its original source")
        if start < previous_end:
            raise ValueError("token spans must be ordered and nonoverlapping")
        if any(character.isspace() for character in text[start:end]):
            raise ValueError("whitespace is a gap, not token content")
        covered[start:end] = b"\x01" * (end - start)
        previous_end = end
    if any(not character.isspace() and not covered[index] for index, character in enumerate(text)):
        raise ValueError("token spans must cover every non-whitespace source character")


def _features(text: str, index: int, previous: str, previous2: str) -> tuple[str, ...]:
    features = ["bias", f"previous={previous}", f"previous2={previous2}",
                f"history={previous2}/{previous}"]
    for relative in (-2, -1, 0, 1, 2):
        position = index + relative
        character = text[position] if 0 <= position < len(text) else None
        category = unicodedata.category(character) if character is not None else "boundary"
        features.extend((f"char{relative}={character!r}", f"category{relative}={category}"))
    features.extend((f"pair={text[index - 1:index + 1]!r}",
                     f"left3={text[max(0, index - 3):index]!r}",
                     f"right3={text[index:index + 3]!r}",
                     f"category-pair={unicodedata.category(text[index - 1])}/{unicodedata.category(text[index])}",
                     f"previous-pair={previous}/{text[index - 1:index + 1]!r}"))
    return tuple(features)


@dataclass
class Segmenter:
    model: Perceptron = field(default_factory=Perceptron)
    metadata: dict[str, Any] = field(default_factory=dict)
    trained: bool = False

    @classmethod
    def train(
        cls, examples: Iterable[tuple[str, tuple[tuple[int, int], ...]]], *,
        epochs: int = 5, seed: int = 0,
    ) -> Segmenter:
        if type(epochs) is not int or epochs < 1:
            raise ValueError("epochs must be a positive integer")
        if type(seed) is not int:
            raise ValueError("seed must be an integer")
        prepared = []
        for text, spans in examples:
            spans = tuple(tuple(span) for span in spans)
            _validate_spans(text, spans)
            prepared.append((text, frozenset(start for start, _ in spans)))
        if not prepared:
            raise ValueError("training requires source-aligned examples")
        segmenter = cls()
        rng = random.Random(seed)
        order = list(prepared)
        for _ in range(epochs):
            rng.shuffle(order)
            for text, starts in order:
                previous, previous2 = "gap", "gap"
                for index, character in enumerate(text):
                    if character.isspace():
                        previous, previous2 = "gap", "gap"
                        continue
                    if index == 0 or text[index - 1].isspace():
                        previous, previous2 = "start", "gap"
                        continue
                    features = _features(text, index, previous, previous2)
                    guess = segmenter.model.predict(features, LABELS)
                    truth = "start" if index in starts else "join"
                    segmenter.model.update(truth, guess, features)
                    previous2, previous = previous, guess
        segmenter.model.average()
        segmenter.trained = True
        segmenter.metadata = {"algorithm": ALGORITHM, "representation": REPRESENTATION,
                              "training_examples": len(prepared),
                              "training_characters": sum(len(text) for text, _ in prepared),
                              "epochs": epochs, "seed": seed,
                              "score_kind": "uncalibrated"}
        return segmenter

    def segment(
        self, text: str, *, beam_width: int = 4, max_candidates: int = 3,
        max_expansions: int = 100000,
    ) -> SegmentationSearch:
        """Return complete source partitions; budget exhaustion never fabricates one.

        Expansions count generated start/join hypotheses. Boundaries at whitespace
        gaps are structural and require no model expansion. Search completeness
        describes exhaustive decoding, not certainty about the intended words.
        """
        if not isinstance(text, str):
            raise TypeError("segmentation source must be text")
        if not self.trained:
            raise ValueError("segmenter has no trained model; no tokenizer fallback exists")
        for name, value, minimum in (("beam_width", beam_width, 1),
                                      ("max_candidates", max_candidates, 1),
                                      ("max_expansions", max_expansions, 0)):
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        # spans, raw score, previous decision, decision before previous
        beam = [((), 0.0, "gap", "gap")]
        expansions = 0
        truncated = False
        reason = None
        last_nonspace = max((index for index, character in enumerate(text) if not character.isspace()), default=-1)
        reached = -1
        for index, character in enumerate(text):
            if character.isspace():
                beam = [(spans, score, "gap", "gap") for spans, score, _, _ in beam]
                continue
            if index == 0 or text[index - 1].isspace():
                beam = [(spans + ((index, index + 1),), score, "start", "gap") for spans, score, _, _ in beam]
                reached = index
                continue
            extended = []
            exhausted = False
            for spans, score, previous, previous2 in beam:
                scored = self.model.score(_features(text, index, previous, previous2), LABELS)
                for label in sorted(LABELS, key=lambda item: (-scored[item], item)):
                    if expansions >= max_expansions:
                        exhausted = True
                        break
                    expansions += 1
                    next_spans = (spans + ((index, index + 1),) if label == "start" else
                                  spans[:-1] + ((spans[-1][0], index + 1),))
                    extended.append((next_spans, score + scored[label], label, previous))
                if exhausted:
                    break
            extended.sort(key=lambda item: (-item[1], item[0]))
            if len(extended) > beam_width:
                truncated, reason = True, "beam_pruned"
            beam = extended[:beam_width]
            reached = index if beam else reached
            if exhausted:
                truncated, reason = True, "budget_exhausted"
                break
        candidates = tuple(SegmentationCandidate(spans, score) for spans, score, _, _ in beam) if reached == last_nonspace else ()
        if len(candidates) > max_candidates:
            truncated = True
            reason = reason or "candidate_limit"
        retained = candidates[:max_candidates]
        return SegmentationSearch(retained, expansions, truncated, not truncated,
                                  reason or ("no_complete_candidates" if not retained else None))


def save_model(path: str | Path, model: Segmenter, metadata: Mapping[str, Any] | None = None) -> None:
    """Write a versioned JSON artifact; no executable pickle or implicit defaults."""
    if not isinstance(model, Segmenter) or not model.trained:
        raise ValueError("only a trained Segmenter can be saved")
    merged_metadata = {**model.metadata, **dict(metadata or {})}
    document = {"version": 1, "algorithm": ALGORITHM, "representation": REPRESENTATION,
                "labels": list(LABELS), "weights": dict(model.model.weights),
                "metadata": merged_metadata}
    # Validate before touching the target, including nonfinite/corrupt weights.
    _from_document(document)
    encoded = json.dumps(document, ensure_ascii=True, sort_keys=True, allow_nan=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _from_document(document: Any) -> Segmenter:
    if not isinstance(document, dict) or type(document.get("version")) is not int or document["version"] != 1:
        raise ValueError("unsupported segmentation artifact version")
    if set(document) != {"version", "algorithm", "representation", "labels", "weights", "metadata"}:
        raise ValueError("invalid segmentation artifact fields")
    if document.get("algorithm") != ALGORITHM or document.get("representation") != REPRESENTATION:
        raise ValueError("unsupported segmentation model or span representation")
    if document.get("labels") != list(LABELS):
        raise ValueError("invalid segmentation boundary labels")
    weights, metadata = document.get("weights"), document.get("metadata")
    if not isinstance(weights, dict) or not isinstance(metadata, dict):
        raise ValueError("segmentation artifact requires weights and metadata mappings")
    model = Perceptron()
    for feature, column in weights.items():
        if not isinstance(feature, str) or not isinstance(column, dict):
            raise ValueError("invalid segmentation feature weights")
        clean = {}
        for label, weight in column.items():
            if label not in LABELS or type(weight) not in (int, float) or not math.isfinite(weight):
                raise ValueError("invalid or nonfinite segmentation weight")
            clean[label] = float(weight)
        model.weights[feature] = clean
    return Segmenter(model=model, metadata=dict(metadata), trained=True)


def load_model(path: str | Path = DEFAULT_MODEL_PATH) -> Segmenter:
    """Load an explicit artifact or fail; no legacy tokenizer is consulted."""
    path = Path(path)
    raw = path.read_bytes()
    try:
        def reject_constant(value: str) -> None:
            raise ValueError(f"nonfinite JSON number {value}")
        document = json.loads(raw.decode("utf-8"), parse_constant=reject_constant)
        model = _from_document(document)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"invalid segmentation artifact {path}: {exc}") from exc
    model.metadata.update(artifact_sha256=hashlib.sha256(raw).hexdigest(), artifact_path=str(path.resolve()))
    return model
