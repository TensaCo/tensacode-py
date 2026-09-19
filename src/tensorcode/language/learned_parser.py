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
    proposals = parser.parse_candidates(words, tags)  # retained legal trees, not a semantic choice

Historical repaired-decoder measurements live only in
``eval.parsing.legacy_baseline``; production inference never fabricates root links.
"""

from __future__ import annotations

import pickle
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generic, Iterable, Sequence, TypeVar

from .treebank import Sentence


Candidate = TypeVar("Candidate")


@dataclass(frozen=True)
class CandidateSearch(Generic[Candidate]):
    """Bounded alternatives, with search coverage distinct from tree completeness.

    ``complete`` means the learned search space was exhausted without beam,
    output, or expansion pruning. It does not establish semantic correctness.
    Every returned candidate is structurally complete even when search truncated.
    Scores are model-internal uncalibrated sums, never probabilities.
    """
    candidates: tuple[Candidate, ...]
    complete: bool
    truncated: bool
    expansions: int
    reason: str | None = None


@dataclass(frozen=True)
class TagCandidate:
    tags: tuple[str, ...]
    score: float
    score_kind: str = "uncalibrated"
    provenance: tuple[str, ...] = ("perceptron",)
    lexical_positions: tuple[int, ...] = ()


@dataclass(frozen=True)
class ParseCandidate:
    heads: dict[int, int]
    labels: dict[int, str]
    score: float
    transitions: tuple[str, ...] = ()
    score_kind: str = "uncalibrated"
    search_score: float | None = None
    ranking: str = "raw"
    provenance: tuple[str, ...] = ("bounded-search",)


def _limits(beam_width: int, max_candidates: int, max_expansions: int) -> None:
    for name, value, minimum in (("beam_width", beam_width, 1),
                                  ("max_candidates", max_candidates, 1),
                                  ("max_expansions", max_expansions, 0)):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")


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
        """Legacy greedy training/evaluation baseline, not interpretation authority."""
        out: list[str] = []
        prev, prev2 = "<s>", "<s2>"
        for i, word in enumerate(words):
            settled = self.known.get(word.lower())
            t = settled or self.model.predict(tag_features(i, words, prev, prev2), self.tags)
            out.append(t)
            prev, prev2 = t, prev
        return out

    def greedy_candidate(self, words: Sequence[str]) -> TagCandidate | None:
        """Retain the old mixed learned-lexicon/perceptron path as a hypothesis.

        Its score excludes the strength of lexical evidence, so it is not a
        calibrated or combined ranking against perceptron-only alternatives.
        Lexical positions are explicit and zero-based. No authored word rules
        are introduced. Ties match the historical tag() evaluation baseline.
        """
        labels = tuple(sorted(set(self.tags)))
        tags = []
        lexical_positions = []
        score = 0.0
        for index, word in enumerate(words):
            previous = tags[-1] if tags else "<s>"
            previous2 = tags[-2] if len(tags) > 1 else "<s>" if tags else "<s2>"
            if not labels:
                return None
            scored = self.model.score(tag_features(index, words, previous, previous2), labels)
            lexical = self.known.get(word.lower())
            if lexical in labels:
                tag = lexical
                lexical_positions.append(index)
            else:
                tag = max(labels, key=lambda label: (scored[label], label))
            tags.append(tag)
            score += scored[tag]
        provenance = ("greedy-learned", "training_lexicon", "perceptron_uncovered") if lexical_positions else ("greedy-learned", "perceptron")
        return TagCandidate(tuple(tags), score, provenance=provenance,
                            lexical_positions=tuple(lexical_positions))

    def lexical_candidate(self, words: Sequence[str]) -> TagCandidate | None:
        """Expose the mixed greedy proposal only when learned lexical evidence applies."""
        candidate = self.greedy_candidate(words)
        return candidate if candidate is not None and candidate.lexical_positions else None

    def tag_candidates(
        self, words: Sequence[str], *, beam_width: int = 8,
        max_candidates: int = 8, max_expansions: int = 10000,
    ) -> CandidateSearch[TagCandidate]:
        """Beam decode all learned tag labels using model weights.

        The old ``known`` shortcut is deliberately absent: one tag observed in
        training does not exclude another tag here. Cached models trained while
        skipping known words may therefore rank differently from greedy tag().
        Stable lexical tie order is bookkeeping, not an interpretation decision.
        Expansion budget counts generated tag extensions, not input tokens.
        """
        _limits(beam_width, max_candidates, max_expansions)
        labels = tuple(sorted(set(self.tags)))
        beam = [TagCandidate((), 0.0)]
        expansions = 0
        truncated = False
        reason = None
        for index in range(len(words)):
            extended = []
            exhausted = False
            for candidate in beam:
                previous = candidate.tags[-1] if candidate.tags else "<s>"
                previous2 = candidate.tags[-2] if len(candidate.tags) > 1 else "<s>" if candidate.tags else "<s2>"
                scored = self.model.score(tag_features(index, words, previous, previous2), labels)
                for label in sorted(labels, key=lambda item: (-scored[item], item)):
                    if expansions >= max_expansions:
                        exhausted = True
                        break
                    expansions += 1
                    extended.append(TagCandidate(candidate.tags + (label,), candidate.score + scored[label]))
                if exhausted:
                    break
            extended.sort(key=lambda candidate: (-candidate.score, candidate.tags))
            if len(extended) > beam_width:
                truncated, reason = True, "beam_pruned"
            beam = extended[:beam_width]
            if exhausted:
                truncated, reason = True, "budget_exhausted"
                break
            if not beam:
                break
        complete_candidates = [candidate for candidate in beam if len(candidate.tags) == len(words)]
        if len(complete_candidates) > max_candidates:
            truncated = True
            reason = reason or "candidate_limit"
        retained = tuple(complete_candidates[:max_candidates])
        return CandidateSearch(retained, not truncated, truncated, expansions,
                               reason or ("no_complete_candidates" if not retained else None))

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
        return out

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


def _candidate_moves(state: State, inventory: Sequence[str]) -> tuple[str, ...]:
    """Structural arc-eager legality, with a single dependency root.

    Inventory supplies learned relation labels. No transition or label is added
    when a model lacks a legal move; no token outside 1..n can be attached.
    """
    legal = []
    for move in sorted(set(inventory)):
        kind, separator, label = move.partition("|")
        if kind == SHIFT and not separator and state.next <= state.n:
            legal.append(move)
        elif kind == REDUCE and not separator and state.stack and state.stack[-1] != 0 and state.stack[-1] in state.heads:
            legal.append(move)
        elif kind == RIGHT and separator and label and state.next <= state.n and state.stack:
            head = state.stack[-1]
            if head == 0:
                if label == "root" and 0 not in state.heads.values():
                    legal.append(move)
            elif label != "root":
                legal.append(move)
        elif kind == LEFT and separator and label and label != "root" and state.next <= state.n and state.stack:
            dependent = state.stack[-1]
            if dependent != 0 and dependent not in state.heads:
                legal.append(move)
    return tuple(legal)


def _complete_tree(state: State) -> bool:
    tokens = set(range(1, state.n + 1))
    if not state.done or set(state.heads) != tokens or set(state.labels) != tokens:
        return False
    if not tokens:
        return True
    if sum(head == 0 for head in state.heads.values()) != 1:
        return False
    for token, head in state.heads.items():
        if head not in tokens | {0} or head == token:
            return False
        seen = set()
        current = token
        while current != 0:
            if current in seen or current not in state.heads:
                return False
            seen.add(current)
            current = state.heads[current]
    return True


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
        return [m for m in self.moves if m.split("|", 1)[0] in legal]

    def greedy_search(
        self, words: Sequence[str], tags: Sequence[str], *, max_steps: int | None = None,
    ) -> CandidateSearch[ParseCandidate]:
        """Audit the historical greedy decisions without repairing invalid trees.

        This is one retained learned path, not exhaustive search. A decision that
        violates strict legality stops with no candidate. The method never makes
        an alternative move merely to preserve baseline coverage.
        """
        if len(words) != len(tags):
            raise ValueError("words and tags must have the same length")
        limit = 2 * len(words) if max_steps is None else max_steps
        _limits(1, 1, limit)
        state = State(len(words))
        score = 0.0
        transitions = []
        while not state.done:
            if len(transitions) >= limit:
                return CandidateSearch((), False, True, len(transitions), "budget_exhausted")
            labels = self._labels(state)
            if not labels:
                return CandidateSearch((), False, True, len(transitions), "no_complete_candidates")
            scored = self.model.score(parse_features(state, words, tags), labels)
            move = max(labels, key=lambda label: (scored[label], label))
            if move not in _candidate_moves(state, self.moves):
                return CandidateSearch((), False, True, len(transitions), "invalid_greedy_transition")
            kind, _, label = move.partition("|")
            state.apply(kind, label)
            score += scored[move]
            transitions.append(move)
        if not _complete_tree(state):
            return CandidateSearch((), False, True, len(transitions), "incomplete_greedy_tree")
        candidate = ParseCandidate(dict(state.heads), dict(state.labels), score, tuple(transitions),
                                   search_score=None, ranking="greedy-local",
                                   provenance=("greedy-learned-unrepaired",))
        return CandidateSearch((candidate,), not words, bool(words), len(transitions),
                               "greedy_path_only" if words else None)

    def greedy_candidate(
        self, words: Sequence[str], tags: Sequence[str], *, max_steps: int | None = None,
    ) -> ParseCandidate | None:
        """A valid unrepaired greedy tree, or None; greedy_search retains why."""
        result = self.greedy_search(words, tags, max_steps=max_steps)
        return result.candidates[0] if result.candidates else None

    def parse_candidates(
        self, words: Sequence[str], tags: Sequence[str], *, beam_width: int = 16,
        max_candidates: int = 8, max_expansions: int = 10000, ranking: str = "raw",
    ) -> CandidateSearch[ParseCandidate]:
        """Retain complete legal dependency trees from a bounded learned beam.

        Unlike the greedy evaluation baseline, this API never falls back to an
        illegal move or attaches leftover tokens to root. Dead ends remain failed
        hypotheses. Tag scores are separate and are not added to parse scores.
        Expansion budget counts generated legal successor states. ``raw`` ranks
        by summed transition scores. Experimental ``local_margin`` ranks by the
        sum of each chosen score minus that state's highest legal score; raw
        scores remain separately retained. Neither score is a probability.
        """
        _limits(beam_width, max_candidates, max_expansions)
        if ranking not in ("raw", "local_margin"):
            raise ValueError("ranking must be raw or local_margin")
        if len(words) != len(tags):
            raise ValueError("words and tags must have the same length")
        beam = [(State(len(words)), 0.0, 0.0, ())]
        finished = {}
        expansions = 0
        truncated = False
        reason = None
        while beam:
            extended = {}
            exhausted = False
            for state, score, search_score, transitions in beam:
                if state.done:
                    if _complete_tree(state):
                        key = (tuple(sorted(state.heads.items())), tuple(sorted(state.labels.items())))
                        candidate = ParseCandidate(dict(state.heads), dict(state.labels), score, transitions,
                                                   search_score=search_score, ranking=ranking)
                        previous = finished.get(key)
                        if previous is None or (-search_score, transitions) < (-previous.search_score, previous.transitions):
                            finished[key] = candidate
                    continue
                legal = _candidate_moves(state, self.moves)
                scored = self.model.score(parse_features(state, words, tags), legal)
                offset = max(scored.values(), default=0.0) if ranking == "local_margin" else 0.0
                for move in sorted(legal, key=lambda item: (-scored[item], item)):
                    if expansions >= max_expansions:
                        exhausted = True
                        break
                    expansions += 1
                    successor = State(state.n, list(state.stack), state.next,
                                      dict(state.heads), dict(state.labels),
                                      defaultdict(list, {key: list(value) for key, value in state.children.items()}))
                    kind, _, label = move.partition("|")
                    successor.apply(kind, label)
                    next_score, history = score + scored[move], transitions + (move,)
                    next_search_score = search_score + scored[move] - offset
                    if successor.done:
                        if _complete_tree(successor):
                            key = (tuple(sorted(successor.heads.items())), tuple(sorted(successor.labels.items())))
                            candidate = ParseCandidate(dict(successor.heads), dict(successor.labels), next_score, history,
                                                       search_score=next_search_score, ranking=ranking)
                            previous = finished.get(key)
                            if previous is None or (-next_search_score, history) < (-previous.search_score, previous.transitions):
                                finished[key] = candidate
                        continue
                    key = (tuple(successor.stack), successor.next,
                           tuple(sorted(successor.heads.items())), tuple(sorted(successor.labels.items())))
                    previous = extended.get(key)
                    if previous is None or (-next_search_score, history) < (-previous[2], previous[3]):
                        extended[key] = (successor, next_score, next_search_score, history)
                if exhausted:
                    break
            ordered = sorted(extended.values(), key=lambda item: (-item[2], item[3]))
            if len(ordered) > beam_width:
                truncated, reason = True, "beam_pruned"
            beam = ordered[:beam_width]
            if exhausted:
                truncated, reason = True, "budget_exhausted"
                break
        ordered_candidates = sorted(finished.values(), key=lambda candidate: (
            -candidate.search_score, tuple(sorted(candidate.heads.items())), tuple(sorted(candidate.labels.items()))))
        if len(ordered_candidates) > max_candidates:
            truncated = True
            reason = reason or "candidate_limit"
        retained = tuple(ordered_candidates[:max_candidates])
        return CandidateSearch(retained, not truncated, truncated, expansions,
                               reason or ("no_complete_candidates" if not retained else None))

    def train(self, sentences: Sequence[Sentence], tagger: Tagger, *, epochs: int = 8, seed: int = 0) -> None:
        """Reproduce the historical static-oracle training updates.

        The local fallback below is retained only for training old artifacts;
        it is not available through an inference method. Future retraining can
        change oracle handling with independently evaluated model provenance.
        """
        def training_legal(state: State) -> list[str]:
            return state.legal() or [SHIFT]

        def training_labels(state: State) -> list[str]:
            legal = set(training_legal(state))
            return [move for move in self.moves if move.split("|", 1)[0] in legal] or list(self.moves)

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
                    if truth not in training_labels(state):
                        truth = SHIFT if SHIFT in training_legal(state) else training_legal(state)[0]
                        kind, label = truth, "dep"
                    guess = self.model.predict(feats, training_labels(state))
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


def save(path: Path, tagger: Tagger, parser: Parser) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps({"tagger": tagger, "parser": parser}, protocol=pickle.HIGHEST_PROTOCOL))


def load_model(path: Path) -> tuple[Tagger, Parser] | None:
    if not path.exists():
        return None
    got = pickle.loads(path.read_bytes())
    return got["tagger"], got["parser"]
