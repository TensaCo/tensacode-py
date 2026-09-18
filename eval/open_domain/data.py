"""Public open-domain benchmarks, loaded from local caches with fixed sampling.

Nothing here touches the network at run time: `scratchpad/open_domain/*.jsonl` is
written once by the fetch step, and HotpotQA comes from the parquet already in the
data directory. Samples are drawn with a fixed seed so all three arms see identical
items, and the item ids are recorded in the results.
"""

from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

SP = Path(os.environ.get("TENSACODE_SCRATCH", "/tmp/claude-1000/-home-brandonin-Documents-tensacode-tensacode-python/c572c14b-5662-4c07-8a7a-1ba7821d2bfa/scratchpad"))
OD = SP / "open_domain"
DATA = Path(os.environ.get("TENSACODE_DATA", SP / "data"))


@dataclass
class Item:
    """One benchmark question. `gold` is the public label; `unanswerable` marks SQuAD 2.0 negatives."""

    id: str
    question: str
    gold: list[str]                       # accepted answer strings (empty when unanswerable)
    passages: list[tuple[str, str]] = field(default_factory=list)  # (title, sentence)
    options: dict[str, str] = field(default_factory=dict)          # multiple choice
    unanswerable: bool = False
    supporting: list[tuple[str, int]] = field(default_factory=list)  # gold (title, sentence index)


_SENT = re.compile(r"(?<=[.!?])\s+")


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT.split(text) if s.strip()]


def squad2(n: int, seed: int = 0) -> list[Item]:
    rows = [json.loads(l) for l in (OD / "squad2_val.jsonl").read_text().splitlines()]
    random.Random(seed).shuffle(rows)
    out = []
    for r in rows[:n]:
        answers = list(dict.fromkeys(r["answers"]["text"]))
        out.append(Item(
            id=r["id"], question=r["question"], gold=answers,
            passages=[(r["title"], s) for s in _sentences(r["context"])],
            unanswerable=not answers,
        ))
    return out


def gsm8k(n: int, seed: int = 0) -> list[Item]:
    rows = [json.loads(l) for l in (OD / "gsm8k_test.jsonl").read_text().splitlines()]
    random.Random(seed).shuffle(rows)
    out = []
    for i, r in enumerate(rows[:n]):
        final = r["answer"].split("####")[-1].strip().replace(",", "")
        out.append(Item(id=f"gsm8k-{i}", question=r["question"], gold=[final]))
    return out


def arc_easy(n: int, seed: int = 0) -> list[Item]:
    rows = [json.loads(l) for l in (OD / "arc_easy_test.jsonl").read_text().splitlines()]
    random.Random(seed).shuffle(rows)
    out = []
    for r in rows[:n]:
        opts = dict(zip(r["choices"]["label"], r["choices"]["text"]))
        if r["answerKey"] not in opts:
            continue
        out.append(Item(id=r["id"], question=r["question"], gold=[r["answerKey"]], options=opts))
    return out[:n]


def hotpot(n: int, seed: int = 0) -> list[Item]:
    import pyarrow.parquet as pq

    table = pq.read_table(DATA / "hotpot_distractor_validation.parquet")
    rows = table.to_pylist()
    random.Random(seed).shuffle(rows)
    out = []
    for r in rows[:n]:
        ctx = r["context"]
        titles, sents = list(ctx["title"]), [list(s) for s in ctx["sentences"]]
        passages = [(t, s.strip()) for t, ss in zip(titles, sents) for s in ss if s.strip()]
        sup = r["supporting_facts"]
        supporting = list(zip(list(sup["title"]), [int(i) for i in sup["sent_id"]]))
        out.append(Item(id=r["id"], question=r["question"], gold=[r["answer"]], passages=passages, supporting=supporting))
    return out


LOADERS = {"squad2": squad2, "gsm8k": gsm8k, "arc_easy": arc_easy, "hotpot": hotpot}
