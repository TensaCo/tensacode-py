"""HotpotQA train/validation loading for selection work.

The train split is the official one (downloaded once to the data directory), so selector
labels are the dataset's own supporting facts and the evaluation items are never trained on.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

from eval.open_domain.data import DATA, Item

TRAIN = DATA / "hotpot_distractor_train_shard0.parquet"
VALID = DATA / "hotpot_distractor_validation.parquet"


@dataclass
class Sentence:
    title: str
    text: str
    gold: bool
    index: int = 0


@dataclass
class Example:
    """One question with every candidate sentence, each marked gold or not."""

    id: str
    question: str
    answer: str
    kind: str
    sentences: list[Sentence] = field(default_factory=list)

    @property
    def n_gold(self) -> int:
        return sum(s.gold for s in self.sentences)


def _rows(path: Path, n: int | None, seed: int) -> list[dict]:
    import pyarrow.parquet as pq

    rows = pq.read_table(path).to_pylist()
    random.Random(seed).shuffle(rows)
    return rows if n is None else rows[:n]


def examples(path: Path, n: int | None = None, seed: int = 0) -> list[Example]:
    out = []
    for r in _rows(path, n, seed):
        ctx = r["context"]
        titles, sents = list(ctx["title"]), [list(s) for s in ctx["sentences"]]
        sup = r["supporting_facts"]
        wanted = set(zip(list(sup["title"]), [int(i) for i in sup["sent_id"]]))
        ex = Example(id=r["id"], question=r["question"], answer=r["answer"], kind=r.get("type", ""))
        for t, ss in zip(titles, sents):
            for i, s in enumerate(ss):
                if s.strip():
                    ex.sentences.append(Sentence(t, s.strip(), (t, i) in wanted, i))
        if ex.n_gold:
            out.append(ex)
    return out


def train(n: int | None = None, seed: int = 0) -> list[Example]:
    return examples(TRAIN, n, seed)


def validation(n: int | None = None, seed: int = 0) -> list[Example]:
    return examples(VALID, n, seed)


def eval_items(n: int = 300, seed: int = 0) -> list[Item]:
    """The same 300 items every other arm in this repo reports on."""
    from eval.open_domain.data import hotpot

    return hotpot(n, seed=seed)
