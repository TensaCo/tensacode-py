"""Graders (public labels only), floors, random controls and Wilson intervals."""

from __future__ import annotations

import math
import random
import re
import string
from collections import Counter

import tensorcode as tc

from .data import Item

ARTICLES = re.compile(r"\b(a|an|the)\b")


def normalize(s: str) -> str:
    """SQuAD's official normalization: lowercase, strip punctuation, articles and extra space."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = ARTICLES.sub(" ", s)
    return " ".join(s.split())


def em(pred: str, golds: list[str]) -> bool:
    return any(normalize(pred) == normalize(g) for g in golds)


def f1(pred: str, golds: list[str]) -> float:
    best = 0.0
    p = normalize(pred).split()
    for g in golds:
        t = normalize(g).split()
        if not p or not t:
            best = max(best, float(p == t))
            continue
        common = Counter(p) & Counter(t)
        same = sum(common.values())
        if same == 0:
            continue
        prec, rec = same / len(p), same / len(t)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best


def numeric_match(pred: str, golds: list[str]) -> bool:
    m = re.findall(r"-?\d[\d,]*\.?\d*", pred.replace("$", ""))
    if not m:
        return False
    try:
        got = float(m[-1].replace(",", ""))
        want = float(golds[0].replace(",", ""))
    except ValueError:
        return False
    return abs(got - want) < 1e-6


def grade(benchmark: str, item: Item, pred: str | tc.Unknown) -> dict:
    """Returns attempted / correct / f1, using only the dataset's own labels.

    SQuAD 2.0 is the one benchmark where abstention is itself gradeable: on an
    unanswerable question, `Unknown` is the right answer and any string is wrong.
    """
    abstained = isinstance(pred, tc.Unknown)
    if benchmark == "squad2":
        if item.unanswerable:
            return {"attempted": not abstained, "correct": abstained, "f1": float(abstained),
                    "kind": "unanswerable", "abstained": abstained}
        if abstained:
            return {"attempted": False, "correct": False, "f1": 0.0, "kind": "answerable", "abstained": True}
        return {"attempted": True, "correct": em(pred, item.gold), "f1": f1(pred, item.gold),
                "kind": "answerable", "abstained": False}
    if benchmark == "gsm8k":
        if abstained:
            return {"attempted": False, "correct": False, "f1": 0.0, "abstained": True}
        return {"attempted": True, "correct": numeric_match(pred, item.gold), "f1": 0.0, "abstained": False}
    if benchmark == "arc_easy":
        if abstained:
            return {"attempted": False, "correct": False, "f1": 0.0, "abstained": True}
        return {"attempted": True, "correct": str(pred).strip().upper()[:1] == item.gold[0], "f1": 0.0, "abstained": False}
    if benchmark == "hotpot":
        if abstained:
            return {"attempted": False, "correct": False, "f1": 0.0, "abstained": True}
        return {"attempted": True, "correct": em(pred, item.gold), "f1": f1(pred, item.gold), "abstained": False}
    raise ValueError(benchmark)


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - h) / d), min(1.0, (c + h) / d))


def floors(benchmark: str, items: list[Item], seed: int = 0) -> dict:
    """Majority / frequency floor and a random control, per benchmark."""
    r = random.Random(seed)
    if benchmark == "arc_easy":
        rand = sum(r.choice(list(i.options)) == i.gold[0] for i in items) / len(items)
        first = sum(sorted(i.options)[0] == i.gold[0] for i in items) / len(items)
        return {"random_choice": round(rand, 4), "always_first_option": round(first, 4),
                "note": "4-way choice: chance is 0.25"}
    if benchmark == "squad2":
        share_unans = sum(i.unanswerable for i in items) / len(items)
        return {"always_abstain": round(share_unans, 4),
                "always_answer_random_span": round(sum(
                    em(r.choice([s for _, s in i.passages]).split()[0] if i.passages else "", i.gold) for i in items) / len(items), 4),
                "note": f"{share_unans:.1%} of the sample is unanswerable, so 'always abstain' scores that much"}
    if benchmark == "gsm8k":
        pick = [float(x) for i in items for x in re.findall(r"\d+", i.question)[:1]] or [0.0]
        rand = sum(numeric_match(str(int(r.choice(pick))), i.gold) for i in items) / len(items)
        return {"random_number_from_the_question": round(rand, 4), "note": "free-form numeric answer: chance is ~0"}
    if benchmark == "hotpot":
        rand = sum(em(r.choice([s for _, s in i.passages]).split()[0] if i.passages else "", i.gold) for i in items) / len(items)
        return {"random_first_word_of_a_random_sentence": round(rand, 4), "note": "free-form span: chance is ~0"}
    return {}


def supporting_overlap(item: Item, used: list[tuple[str, str]]) -> dict | None:
    """For HotpotQA: did the evidence the arm actually cited contain the gold supporting facts?"""
    if not item.supporting:
        return None
    gold_sents = set()
    by_title: dict[str, list[str]] = {}
    for t, s in item.passages:
        by_title.setdefault(t, []).append(s)
    for title, idx in item.supporting:
        sents = by_title.get(title, [])
        if 0 <= idx < len(sents):
            gold_sents.add(normalize(sents[idx]))
    if not gold_sents:
        return None
    cited = {normalize(s) for _, s in used}
    hit = len(gold_sents & cited)
    return {"gold": len(gold_sents), "cited": len(cited), "hit": hit, "all_gold_cited": hit == len(gold_sents)}
