"""Historical repaired decoder, retained exclusively for evaluation comparison.

This code reproduces published baseline measurements, including illegal-move
fallback and root attachment repair. It is not an agent interpretation provider
and must not be imported by production code. Candidate inference lives in
``tensorcode.language.learned_parser`` and refuses these incomplete structures.
"""
from __future__ import annotations

from typing import Sequence

from tensorcode.language.learned_parser import Parser, Tagger, State, SHIFT, parse_features
from tensorcode.language.treebank import Sentence


def _legacy_labels(parser: Parser, state: State) -> list[str]:
    legal = set(state.legal() or [SHIFT])
    return [move for move in parser.moves if move.split("|", 1)[0] in legal] or list(parser.moves)


def parse(parser: Parser, words: Sequence[str], tags: Sequence[str]) -> tuple[dict[int, int], dict[int, str]]:
    """Legacy greedy evaluation baseline, including historical repair behavior."""
    state = State(len(words))
    guard = 0
    while not state.done and guard < 4 * len(words) + 10:
        guard += 1
        feats = parse_features(state, words, tags)
        move = parser.model.predict(feats, _legacy_labels(parser, state))
        kind, _, label = move.partition("|")
        state.apply(kind, label or "dep")
    for i in range(1, len(words) + 1):  # anything unattached hangs off the root
        state.heads.setdefault(i, 0)
        state.labels.setdefault(i, "dep")
    return state.heads, state.labels


def scores(tagger: Tagger, parser: Parser, sentences: Sequence[Sentence]) -> dict:
    """Tagging accuracy, and unlabelled/labelled attachment scores (punctuation excluded)."""
    tag_right = tag_total = uas = las = total = 0
    for s in sentences:
        words = [t.form for t in s]
        gold_tags = [t.upos for t in s]
        tags = tagger.tag(words)
        tag_right += sum(1 for a, b in zip(tags, gold_tags) if a == b)
        tag_total += len(s)
        heads, labels = parse(parser, words, tags)
        for t in s:
            if t.upos == "PUNCT":
                continue
            total += 1
            if heads.get(t.id) == t.head:
                uas += 1
                if labels.get(t.id) == t.deprel:
                    las += 1
    return {"tagging_accuracy": round(tag_right / max(1, tag_total), 4),
            "uas": round(uas / max(1, total), 4), "las": round(las / max(1, total), 4),
            "sentences": len(sentences), "tokens_scored": total}

