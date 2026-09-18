"""A part-of-speech tagger and a dependency parser, learned from a treebank.

Both are averaged perceptrons over sparse symbolic features: the model is a dictionary
from (feature, label) to a weight, trained by making a prediction, and when it is wrong,
adding 1 to the right label's features and subtracting 1 from the wrong one's. There is
no network, no gradient and no matrix: scoring a word is summing a few dozen floats.
(Collins 2002 for the averaged perceptron; Nivre 2003 for arc-eager parsing.)

Why this exists: the hand-written grammar in ``english.py`` is knowledge in code, with
weights I chose. Here the same knowledge — what a word can be, what attaches to what —
is *estimated from a treebank*, and measured on its held-out split, so "does it parse
English?" has an answer rather than an impression.

    tagger, parser = train(load("train"))
    scores(tagger, parser, load("test"))   # tagging accuracy, unlabelled and labelled attachment
"""

from __future__ import annotations

import pickle
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from .treebank import Sentence


class Perceptron:
    """Averaged perceptron: weights per (feature, label), averaged over all updates."""

    def __init__(self) -> None:
        self.weights: dict[str, dict[str, float]] = defaultdict(dict)
        self.totals: dict[tuple[str, str], float] = defaultdict(float)
        self.stamps: dict[tuple[str, str], int] = defaultdict(int)
        self.i = 0

    def score(self, features: Iterable[str], labels: Sequence[str]) -> dict[str, float]:
        out = {label: 0.0 for label in labels}
        for f in features:
            column = self.weights.get(f)
            if not column:
                continue
            for label, w in column.items():
                if label in out:
                    out[label] += w
        return out

    def predict(self, features: Sequence[str], labels: Sequence[str]) -> str:
        scored = self.score(features, labels)
        return max(labels, key=lambda label: (scored[label], label))

    def update(self, truth: str, guess: str, features: Iterable[str]) -> None:
        self.i += 1
        if truth == guess:
            return
        for f in features:
            column = self.weights.setdefault(f, {})
            for label, sign in ((truth, 1.0), (guess, -1.0)):
                key = (f, label)
                w = column.get(label, 0.0)
                self.totals[key] += (self.i - self.stamps[key]) * w
                self.stamps[key] = self.i
                column[label] = w + sign

    def average(self) -> None:
        for f, column in self.weights.items():
            for label, w in list(column.items()):
                key = (f, label)
                total = self.totals[key] + (self.i - self.stamps[key]) * w
                averaged = round(total / self.i, 4)
                if averaged:
                    column[label] = averaged
                else:
                    del column[label]


# ------------------------------------------------------------------------- tagging


def shape(word: str) -> str:
    if word.isdigit():
        return "0"
    if any(c.isdigit() for c in word):
        return "alnum"
    if word[:1].isupper():
        return "Xx" if word[1:].islower() else "XX"
    return "x"


def tag_features(i: int, words: Sequence[str], prev: str, prev2: str) -> list[str]:
    w = words[i].lower()
    return [
        "b", f"w={w}", f"suf3={w[-3:]}", f"suf2={w[-2:]}", f"pre1={w[:1]}", f"shape={shape(words[i])}",
        f"t-1={prev}", f"t-2={prev2}", f"t-1t-2={prev} {prev2}",
        f"w-1={words[i - 1].lower() if i else '<s>'}", f"w+1={words[i + 1].lower() if i + 1 < len(words) else '</s>'}",
        f"t-1 w={prev} {w}", f"suf3 t-1={w[-3:]} {prev}",
    ]


@dataclass
class Tagger:
    model: Perceptron = field(default_factory=Perceptron)
    tags: tuple[str, ...] = ()
    known: dict[str, str] = field(default_factory=dict)  # words with one tag in training: settled

    def tag(self, words: Sequence[str]) -> list[str]:
        out: list[str] = []
        prev, prev2 = "<s>", "<s2>"
        for i, word in enumerate(words):
            settled = self.known.get(word.lower())
            t = settled or self.model.predict(tag_features(i, words, prev, prev2), self.tags)
            out.append(t)
            prev, prev2 = t, prev
        return out

    def train(self, sentences: Sequence[Sentence], *, epochs: int = 5, seed: int = 0) -> None:
        counts: dict[str, set[str]] = defaultdict(set)
        for s in sentences:
            for t in s:
                counts[t.form.lower()].add(t.upos)
        self.known = {w: next(iter(ts)) for w, ts in counts.items() if len(ts) == 1 and w.isalpha()}
        self.tags = tuple(sorted({t.upos for s in sentences for t in s}))
        rng = random.Random(seed)
        order = list(sentences)
        for _ in range(epochs):
            rng.shuffle(order)
            for s in order:
                words = [t.form for t in s]
                prev, prev2 = "<s>", "<s2>"
                for i, token in enumerate(s):
                    settled = self.known.get(words[i].lower())
                    if settled:
                        guess = settled
                    else:
                        feats = tag_features(i, words, prev, prev2)
                        guess = self.model.predict(feats, self.tags)
                        self.model.update(token.upos, guess, feats)
                    prev, prev2 = guess, prev
        self.model.average()


# ------------------------------------------------------------------------- parsing

SHIFT, RIGHT, LEFT, REDUCE = "shift", "right", "left", "reduce"


@dataclass
class State:
    n: int
    stack: list[int] = field(default_factory=lambda: [0])
    next: int = 1
    heads: dict[int, int] = field(default_factory=dict)
    labels: dict[int, str] = field(default_factory=dict)
    children: dict[int, list[int]] = field(default_factory=lambda: defaultdict(list))

    @property
    def done(self) -> bool:
        return self.next > self.n and len(self.stack) <= 1

    def legal(self) -> list[str]:
        out = []
        if self.next <= self.n:
            out.append(SHIFT)
            if self.stack:
                out.append(RIGHT)
                if self.stack[-1] != 0 and self.stack[-1] not in self.heads:
                    out.append(LEFT)
        if self.stack and self.stack[-1] != 0 and self.stack[-1] in self.heads:
            out.append(REDUCE)
        return out or [SHIFT]

    def apply(self, move: str, label: str = "dep") -> None:
        if move == SHIFT:
            self.stack.append(self.next)
            self.next += 1
        elif move == RIGHT:
            head, dependent = self.stack[-1], self.next
            self.heads[dependent], self.labels[dependent] = head, label
            self.children[head].append(dependent)
            self.stack.append(dependent)
            self.next += 1
        elif move == LEFT:
            dependent, head = self.stack.pop(), self.next
            self.heads[dependent], self.labels[dependent] = head, label
            self.children[head].append(dependent)
        elif move == REDUCE:
            self.stack.pop()


def parse_features(state: State, words: Sequence[str], tags: Sequence[str]) -> list[str]:
    def w(i: int) -> str:
        return "<root>" if i == 0 else (words[i - 1].lower() if 1 <= i <= len(words) else "<none>")

    def p(i: int) -> str:
        return "<root>" if i == 0 else (tags[i - 1] if 1 <= i <= len(tags) else "<none>")

    s0 = state.stack[-1] if state.stack else -1
    s1 = state.stack[-2] if len(state.stack) > 1 else -1
    b0, b1, b2 = state.next, state.next + 1, state.next + 2
    kids = state.children.get(s0, [])
    bkids = state.children.get(b0, [])
    s0h = state.heads.get(s0, -1)
    lc = min(kids) if kids else -1
    rc = max(kids) if kids else -1
    blc = min(bkids) if bkids else -1
    return [
        f"s0hp={p(s0h)}", f"s0lcp={p(lc)}", f"s0rcp={p(rc)}", f"b0lcp={p(blc)}",
        f"s0lcl={state.labels.get(lc, '<none>')}", f"s0rcl={state.labels.get(rc, '<none>')}",
        f"s0l={state.labels.get(s0, '<none>')}", f"s0valency={min(5, len(kids))}/{p(s0)}",
        f"s0p s0rcp b0p={p(s0)} {p(rc)} {p(b0)}", f"s0p s0lcp b0p={p(s0)} {p(lc)} {p(b0)}",
        f"s0hp s0p b0p={p(s0h)} {p(s0)} {p(b0)}", f"s1w={w(s1)}", f"b1w={w(b1)}",
        f"s0w s1w={w(s0)} {w(s1)}", f"s0p s1p={p(s0)} {p(s1)}", f"b0w b1w={w(b0)} {w(b1)}",
        "b",
        f"s0w={w(s0)}", f"s0p={p(s0)}", f"s0wp={w(s0)}/{p(s0)}",
        f"b0w={w(b0)}", f"b0p={p(b0)}", f"b0wp={w(b0)}/{p(b0)}",
        f"b1p={p(b1)}", f"b2p={p(b2)}", f"s1p={p(s1)}",
        f"s0p b0p={p(s0)} {p(b0)}", f"s0w b0w={w(s0)} {w(b0)}", f"s0p b0w={p(s0)} {w(b0)}", f"s0w b0p={w(s0)} {p(b0)}",
        f"s0p b0p b1p={p(s0)} {p(b0)} {p(b1)}", f"s1p s0p b0p={p(s1)} {p(s0)} {p(b0)}",
        f"dist={min(9, b0 - s0) if s0 >= 0 else -1}", f"s0kids={len(kids)}",
        f"s0lc={state.labels.get(kids[0], '<none>') if kids else '<none>'}",
        f"s0rc={state.labels.get(kids[-1], '<none>') if kids else '<none>'}",
        f"stack={min(4, len(state.stack))}",
    ]


def oracle(state: State, heads: dict[int, int], labels: dict[int, str]) -> tuple[str, str]:
    """The move a correct parse makes here (static oracle over the gold tree)."""
    s0, b0 = (state.stack[-1] if state.stack else -1), state.next
    if s0 > 0 and heads.get(s0) == b0 and s0 not in state.heads:
        return LEFT, labels[s0]
    if b0 <= state.n and heads.get(b0) == s0:
        return RIGHT, labels[b0]
    if s0 > 0 and s0 in state.heads:
        below = state.stack[:-1]
        # reduce when what comes next belongs to something already on the stack, so s0 is done
        if heads.get(b0) in below or any(heads.get(k) == b0 for k in below):
            return REDUCE, "dep"
    return SHIFT, "dep"


@dataclass
class Parser:
    model: Perceptron = field(default_factory=Perceptron)
    moves: tuple[str, ...] = ()

    def _labels(self, state: State) -> list[str]:
        legal = set(state.legal())
        return [m for m in self.moves if m.split("|", 1)[0] in legal] or list(self.moves)

    def parse(self, words: Sequence[str], tags: Sequence[str]) -> tuple[dict[int, int], dict[int, str]]:
        state = State(len(words))
        guard = 0
        while not state.done and guard < 4 * len(words) + 10:
            guard += 1
            feats = parse_features(state, words, tags)
            move = self.model.predict(feats, self._labels(state))
            kind, _, label = move.partition("|")
            state.apply(kind, label or "dep")
        for i in range(1, len(words) + 1):  # anything unattached hangs off the root
            state.heads.setdefault(i, 0)
            state.labels.setdefault(i, "dep")
        return state.heads, state.labels

    def train(self, sentences: Sequence[Sentence], tagger: Tagger, *, epochs: int = 8, seed: int = 0) -> None:
        self.moves = tuple(sorted({f"{k}|{t.deprel}" for s in sentences for t in s for k in (RIGHT, LEFT)}
                                  | {SHIFT, REDUCE}))
        rng = random.Random(seed)
        order = list(sentences)
        for _ in range(epochs):
            rng.shuffle(order)
            for s in order:
                words = [t.form for t in s]
                # the parser is trained on the tags it will actually be given, not gold ones:
                # otherwise it learns to trust tags that are 6% wrong at use time
                tags = tagger.tag(words)
                heads = {t.id: t.head for t in s}
                labels = {t.id: t.deprel for t in s}
                state = State(len(words))
                guard = 0
                while not state.done and guard < 4 * len(words) + 10:
                    guard += 1
                    feats = parse_features(state, words, tags)
                    kind, label = oracle(state, heads, labels)
                    truth = kind if kind in (SHIFT, REDUCE) else f"{kind}|{label}"
                    if truth not in self._labels(state):
                        truth = SHIFT if SHIFT in state.legal() else state.legal()[0]
                        kind, label = truth, "dep"
                    guess = self.model.predict(feats, self._labels(state))
                    self.model.update(truth, guess, feats)
                    gk, _, gl = truth.partition("|")
                    state.apply(gk, gl or "dep")
        self.model.average()


def train(sentences: Sequence[Sentence], *, tag_epochs: int = 5, parse_epochs: int = 8) -> tuple[Tagger, Parser]:
    tagger = Tagger()
    tagger.train(sentences, epochs=tag_epochs)
    parser = Parser()
    parser.train(sentences, tagger, epochs=parse_epochs)
    return tagger, parser


def scores(tagger: Tagger, parser: Parser, sentences: Sequence[Sentence]) -> dict:
    """Tagging accuracy, and unlabelled/labelled attachment scores (punctuation excluded)."""
    tag_right = tag_total = uas = las = total = 0
    for s in sentences:
        words = [t.form for t in s]
        gold_tags = [t.upos for t in s]
        tags = tagger.tag(words)
        tag_right += sum(1 for a, b in zip(tags, gold_tags) if a == b)
        tag_total += len(s)
        heads, labels = parser.parse(words, tags)
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


def save(path: Path, tagger: Tagger, parser: Parser) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps({"tagger": tagger, "parser": parser}, protocol=pickle.HIGHEST_PROTOCOL))


def load_model(path: Path) -> tuple[Tagger, Parser] | None:
    if not path.exists():
        return None
    got = pickle.loads(path.read_bytes())
    return got["tagger"], got["parser"]
