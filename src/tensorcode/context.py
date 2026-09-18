"""Deterministic context assembly: redundancy removal and budgeted packing.

Packing is a lossy conversion, so the result says exactly what was dropped and why.
Required evidence is never silently dropped: if it cannot fit, the answer is Unknown.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Generic, Hashable, Sequence, TypeVar

from .outcomes import Score, Unknown

T = TypeVar("T")

_TOKEN = re.compile(r"\w+|[^\w\s]")


def approx_tokens(text: str) -> int:
    """Word-and-punctuation count; a stand-in for a real tokenizer.

    Measured against the Qwen3 tokenizer on 12,187 HotpotQA sentences: 0.82x the true total
    (per-sentence median 0.84, p5 0.64), so it *undercounts*. Pass a real tokenizer as ``cost``
    when a budget is a hard model limit.
    """
    return len(_TOKEN.findall(text))


def shingle_similarity(a: str, b: str, n: int = 3) -> Score:
    """Jaccard overlap of word n-grams. A similarity, not a probability of duplication."""

    def grams(s: str) -> set[tuple[str, ...]]:
        words = s.lower().split()
        return {tuple(words[i : i + n]) for i in range(max(1, len(words) - n + 1))}

    ga, gb = grams(a), grams(b)
    return Score(len(ga & gb) / len(ga | gb) if ga | gb else 0.0, "similarity")


@dataclass(frozen=True)
class Packed(Generic[T]):
    items: tuple[T, ...]
    used: int
    budget: int
    dropped: tuple[tuple[T, str], ...]  # (item, reason)


def dedupe(
    ranked: Sequence[tuple[T, Score]],
    *,
    similarity: Callable[[T, T], Score],
    threshold: float,
    keep: Callable[[T], bool] = lambda item: False,
    key: Callable[[T], Hashable] = id,
) -> tuple[list[tuple[T, Score]], list[tuple[T, str]]]:
    """Walk in rank order; drop an item too similar to one already kept (unless ``keep`` says otherwise)."""
    kept: list[tuple[T, Score]] = []
    dropped: list[tuple[T, str]] = []
    for item, score in ranked:
        dup = next((k for k, _ in kept if similarity(item, k).value >= threshold), None)
        if dup is not None and not keep(item):
            dropped.append((item, f"near-duplicate of {key(dup)!r}"))
        else:
            kept.append((item, score))
    return kept, dropped


def pack(
    ranked: Sequence[tuple[T, Score]],
    *,
    budget: int,
    cost: Callable[[T], int],
    required: Sequence[T] = (),
    key: Callable[[T], Hashable] = id,
    dropped: Sequence[tuple[T, str]] = (),
) -> Packed[T] | Unknown:
    """Required items first, then ranked items greedily while they fit.

    ``dropped`` carries removals made earlier (e.g. by ``dedupe``) into the same report.
    """
    required_keys = {key(r) for r in required}
    chosen: list[T] = list(required)
    used = sum(cost(r) for r in required)
    if used > budget:
        return Unknown("required_evidence_exceeds_budget", f"required items cost {used} > budget {budget}")
    dropped = list(dropped)
    for item, _ in ranked:
        if key(item) in required_keys:
            continue
        c = cost(item)
        if used + c <= budget:
            chosen.append(item)
            used += c
        else:
            dropped.append((item, f"budget: needs {c}, {budget - used} left"))
    return Packed(tuple(chosen), used, budget, tuple(dropped))
