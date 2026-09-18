"""An Earley chart over a feature grammar, then a Viterbi cover for robustness.

Two stages, because they answer different questions.

*The chart* finds every constituent the grammar licenses over every span,
unifying features as it goes and assembling meaning from each production's
semantic spec. Earley (not CKY) so the grammar can stay readable: no
binarisation and left recursion is allowed. Constituents are found at every
position, not only from the sentence start, because the cover needs fragments.

*The cover* decides what the whole utterance means when the grammar does not
span it. An utterance is a sequence of top-level constituents and skipped
tokens, and skipped tokens are charged at a **fitted background unigram** rather
than a hand-chosen skip penalty, so a partial reading and a full reading are
comparable on one scale. That device is taken from ``symbolic-ai-models``
(``symbolic_ai_parsers/parsers/cky_001``), where a constant penalty was noted to
be a free parameter sitting under every parser at once.

What is not recoverable is reported rather than guessed: skipped tokens survive
on the reading, and genuine ambiguity survives as several equal-scoring readings.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Iterable, Mapping, Sequence

from ..outcomes import Score
from .features import Bindings, FVar, ground, unify
from .grammar import ABSENT, Cat, Grammar, Terminal, build_sem
from .semantics import _key_of

#: quoted spans, paths, numbers, words (with internal ' - _ . @ +), then punctuation
_TOKEN = re.compile(
    # a single-quoted span may contain a contraction: 'don't forget' is one literal,
    # so an interior apostrophe is allowed when a letter follows it
    r"""'(?:[^']|'(?=[A-Za-z]))*'|"[^"]*"|`[^`]*`|“[^”]*”|‘[^’]*’"""
    r"""|~(?:/[^\s,;:!?]*)?|/[\w.@+\-/]+|\.\.\.|\d+(?:\.\d+)?|[\w][\w'’.\-_@+]*|[^\s\w]"""
)
QUOTES = {"'": "'", '"': '"', "`": "`", "“": "”", "‘": "’"}


#: Clitics are separate words: "what's in it" is "what" + "'s", and a grammar that
#: cannot see the auxiliary cannot read the question.
CLITICS = ("n't", "'s", "'re", "'ll", "'ve", "'m", "'d")


def tokenize(text: str) -> list[str]:
    """Words, numbers, paths and quoted spans; a quoted span stays one token."""
    out: list[str] = []
    for token in _TOKEN.findall(text.strip()):
        if not token.strip():
            continue
        if not is_quoted(token) and "/" not in token:
            # A word may contain a dot ("hi.txt", "3.14") but may not *end* with one:
            # that dot is the end of the sentence. Keeping it attached made "food." a
            # token no lexicon and no open-class pattern could match, so the last word
            # of every sentence entered as an unknown name — invisible here, because a
            # guessed name absorbs anything, and expensive in a caller that counts
            # unknown words.
            stops = ""
            while len(token) > 1 and token.endswith("."):
                token, stops = token[:-1], stops + "."
            if stops:
                out.extend(_split_clitic(token))
                out.extend(stops)
                continue
            for clitic in CLITICS:
                if len(token) > len(clitic) and token.lower().endswith(clitic):
                    out.extend([token[: -len(clitic)], token[-len(clitic):]])
                    break
            else:
                out.append(token)
            continue
        out.append(token)
    return out


def _split_clitic(token: str) -> list[str]:
    for clitic in CLITICS:
        if len(token) > len(clitic) and token.lower().endswith(clitic):
            return [token[: -len(clitic)], token[-len(clitic):]]
    return [token]


def unquote(token: str) -> str:
    if len(token) >= 2 and token[0] in QUOTES and token[-1] == QUOTES[token[0]]:
        return token[1:-1]
    return token


def is_quoted(token: str) -> bool:
    return len(token) >= 2 and token[0] in QUOTES and token[-1] == QUOTES[token[0]]


# ----------------------------------------------------------------- chart nodes


class Node:
    """A completed constituent: what it is, where it is, and what it means.

    A plain slotted class, not a dataclass: the chart consults ``key`` and ``sem_key``
    thousands of times per parse, so both are computed once here rather than through a
    ``cached_property`` descriptor (which was 572 ms of a 3.1 s run) or, as before,
    through ``repr``.
    """

    __slots__ = ("cat", "start", "end", "sem", "feats", "weight", "rule", "children", "words", "sem_key", "key",
                 "serial")

    def __init__(self, cat: str, start: int, end: int, sem: Any, feats: tuple[tuple[str, Any], ...] = (),
                 weight: float = 0.0, rule: str = "", children: tuple["Node", ...] = (),
                 words: tuple[str, ...] = ()) -> None:
        self.cat, self.start, self.end, self.sem = cat, start, end, sem
        self.feats, self.weight, self.rule = feats, weight, rule
        self.children, self.words = children, words
        self.sem_key = _key_of(sem)
        self.key = (cat, start, end, self.sem_key)
        self.serial = 0  # set by the chart, in creation order, for deterministic ties

    def __hash__(self) -> int:
        return hash(self.key)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Node) and self.key == other.key

    def __repr__(self) -> str:
        return f"Node({self.cat}[{self.start}:{self.end}] {self.rule})"

    def features(self) -> dict[str, Any]:
        return dict(self.feats)

    def leaves(self) -> Iterable["Node"]:
        if not self.children:
            yield self
        for child in self.children:
            yield from child.leaves()

    def guesses(self) -> list[tuple[str, str]]:
        """(token, category) for every word that entered on an open-class guess."""
        return [(leaf.words[0], leaf.cat) for leaf in self.leaves()
                if leaf.words and dict(leaf.feats).get("guessed")]

    def tree(self, indent: int = 0) -> str:
        pad = "  " * indent
        rows = [f"{pad}{self.cat}[{self.start}:{self.end}] {self.rule}".rstrip()]
        rows += [c.tree(indent + 1) for c in self.children]
        return "\n".join(rows)


class _Item:
    """A dotted rule with its bindings and the daughters matched so far."""

    __slots__ = ("prod", "dot", "start", "bindings", "children", "weight", "symbol", "done", "key")

    def __init__(self, prod: Any, dot: int, start: int, bindings: tuple[tuple[str, Any], ...],
                 children: tuple[Node, ...], weight: float) -> None:
        self.prod, self.dot, self.start = prod, dot, start
        self.bindings, self.children, self.weight = bindings, children, weight
        self.done = dot >= len(prod.rhs)
        self.symbol = None if self.done else prod.rhs[dot]
        self.key = (prod.name, dot, start, bindings, tuple(c.key for c in children))

    def next_symbol(self) -> Any:
        return self.symbol


def _freeze(bindings: Bindings) -> tuple[tuple[str, Any], ...]:
    return tuple(sorted(bindings.items(), key=lambda kv: kv[0]))


class Chart:
    """Every constituent the grammar licenses, indexed by span and by start.

    Distinct meanings over one span are kept (that is real ambiguity), capped, and
    looked up by content key — a dict, not a scan with string comparisons.
    """

    def __init__(self, cap: int = 8) -> None:
        self.cap = cap
        self.by_span: dict[tuple[int, int, str], dict[Any, Node]] = {}
        self.by_start: dict[tuple[int, str], list[Node]] = {}
        self._serial = 0

    def add(self, node: Node) -> bool:
        self._serial += 1
        node.serial = self._serial
        span = self.by_span.setdefault((node.start, node.end, node.cat), {})
        key = node.sem_key
        seen = span.get(key)
        if seen is not None:
            if node.weight <= seen.weight:
                return False
            span[key] = node
            self._replace(node, seen)
            return True
        if len(span) >= self.cap:
            # lowest weight, and among equal weights the oldest — the same node the
            # previous linear scan dropped, so capping a span still yields the same
            # surviving set rather than one that depends on dict ordering
            worst_key = min(span, key=lambda k: (span[k].weight, span[k].serial))
            worst = span[worst_key]
            if worst.weight >= node.weight:
                return False
            del span[worst_key]
            span[key] = node
            self._replace(node, worst)
            return True
        span[key] = node
        self.by_start.setdefault((node.start, node.cat), []).append(node)
        return True

    def _replace(self, node: Node, dropped: Node) -> None:
        row = self.by_start.setdefault((node.start, node.cat), [])
        for i, other in enumerate(row):
            if other is dropped:
                row[i] = node
                return
        row.append(node)

    def spanning(self, start: int, end: int, cat: str) -> list[Node]:
        return list(self.by_span.get((start, end, cat), {}).values())

    def starting(self, start: int, cats: Iterable[str]) -> list[Node]:
        out: list[Node] = []
        for cat in sorted(set(cats)):
            out.extend(self.by_start.get((start, cat), ()))
        return out

    def size(self) -> int:
        return sum(len(v) for v in self.by_span.values())


def build_chart(grammar: Grammar, tokens: Sequence[str], *, cap: int = 8, max_tokens: int = 40,
                max_items: int = 20000) -> Chart:
    """Earley recognition with feature unification and semantic assembly."""
    n = min(len(tokens), max_tokens)
    barriers = {i for i, t in enumerate(tokens[:n]) if t.lower() in grammar.barriers}
    chart = Chart(cap)
    columns: list[dict[tuple, _Item]] = [dict() for _ in range(n + 1)]
    agendas: list[list[_Item]] = [[] for _ in range(n + 1)]
    # items in a column indexed by the category they are waiting for. Completion used
    # to walk the whole start column and let ``_advance`` reject the mismatches, which
    # is O(items in the column) for every node the chart finds; a clause of any length
    # spends nearly all of it saying no.
    waiting: list[dict[str, list[tuple]]] = [dict() for _ in range(n + 1)]

    def push(column: int, item: _Item) -> None:
        seen = columns[column].get(item.key)
        if seen is None or item.weight > seen.weight:
            columns[column][item.key] = item
            agendas[column].append(item)
            if seen is None and not item.done and type(item.symbol) is Cat:
                # the key, not the item: a later push may replace this item with a
                # better-weighted one under the same key, and completion wants that one
                waiting[column].setdefault(item.symbol.name, []).append(item.key)

    # lexical nodes are pre-terminals: productions never spell words out
    for i in range(n):
        for entry in grammar.entries_for(tokens[i]):
            sem = entry.sem if entry.sem is not None else tokens[i]
            chart.add(Node(entry.cat, i, i + 1, sem, _freeze(dict(entry.features)), entry.weight,
                           f"lex:{entry.word}", (), (tokens[i],)))

    # A constituent may begin at any position, because the cover needs fragments — but
    # only one whose first symbol can actually start there. Seeding every production at
    # every position was ~90 items per column, nearly all of them dead on arrival.
    startable = _startable(grammar, tokens, chart, n)

    for i in range(n + 1):
        for prod in startable[i]:
            push(i, _Item(prod, 0, i, (), (), prod.weight))
        processed: set[tuple] = set()
        while agendas[i] and len(processed) < max_items:
            item = agendas[i].pop()
            if item.key in processed:
                continue
            processed.add(item.key)
            bindings: Bindings = dict(item.bindings)

            if item.done:
                feats = ground(item.prod.lhs.features, bindings)
                try:
                    sem = build_sem(item.prod.sem, [c.sem for c in item.children], [c.words for c in item.children], [c.features() for c in item.children])
                except (IndexError, TypeError, KeyError):
                    continue
                if any(item.start <= b < i for b in barriers) and i - item.start > 1:
                    continue  # no constituent spans a barrier word
                node = Node(item.prod.lhs.name, item.start, i, sem, _freeze(feats), item.weight, item.prod.name,
                            item.children, tuple(w for c in item.children for w in c.words))
                if chart.add(node) and item.start < i:
                    for key in waiting[item.start].get(node.cat, ()):
                        current = columns[item.start].get(key)
                        if current is None:
                            continue
                        advanced = _advance(current, node)
                        if advanced is not None:
                            push(i, advanced)
                continue

            symbol = item.symbol
            if isinstance(symbol, Terminal):
                if i < n and tokens[i].lower() == symbol.word.lower():
                    leaf = Node(f'"{symbol.word}"', i, i + 1, symbol.word, (), 0.0, "terminal", (), (tokens[i],))
                    push(i + 1, _Item(item.prod, item.dot + 1, item.start, item.bindings, item.children + (leaf,), item.weight))
                continue

            assert isinstance(symbol, Cat)
            wanted = ground(symbol.features, bindings)
            for prod in grammar.by_lhs(symbol.name):
                # Predict with *empty* bindings. Carrying the parent's constraints in
                # (or freshening variables per prediction) gives two items the same
                # dotted rule with different binding sets, which defeats Earley's
                # dedupe and makes a left-recursive production loop forever. The
                # parent's constraint is still enforced, in `_advance` on completion.
                if unify(wanted, prod.lhs.features, {}) is None:
                    continue
                push(i, _Item(prod, 0, i, (), (), prod.weight))
            for node in chart.starting(i, [symbol.name]):
                advanced = _advance(item, node)
                if advanced is not None:
                    push(node.end, advanced)
    return chart


def _startable(grammar: Grammar, tokens: Sequence[str], chart: Chart, n: int) -> list[list[Any]]:
    """Productions whose first symbol can begin at each position.

    A production has no empty right-hand side (``production`` refuses one), so it can
    only start where its first symbol can. The set of categories available at a
    position is the lexical ones there, closed under "a production whose first symbol
    is available makes its own category available".
    """
    # the grammar's own order, not a set's: iterating `categories()` made seeding — and
    # therefore which of two equal-scoring readings won — depend on set ordering
    all_prods = list(grammar.productions)
    out: list[list[Any]] = []
    for i in range(n + 1):
        available = {cat for (start, cat) in chart.by_start if start == i}
        word = tokens[i].lower() if i < n else None
        changed = True
        while changed:
            changed = False
            for prod in all_prods:
                if prod.lhs.name in available:
                    continue
                first = prod.rhs[0]
                if isinstance(first, Terminal):
                    if word is not None and first.word.lower() == word:
                        available.add(prod.lhs.name)
                        changed = True
                elif first.name in available:
                    available.add(prod.lhs.name)
                    changed = True
        seeds = []
        for prod in all_prods:
            first = prod.rhs[0]
            if isinstance(first, Terminal):
                if word is not None and first.word.lower() == word:
                    seeds.append(prod)
            elif first.name in available:
                seeds.append(prod)
        out.append(seeds)
    return out


def _advance(item: _Item, node: Node) -> _Item | None:
    """Move an item's dot over a completed constituent, if their features agree."""
    symbol = item.symbol
    if not isinstance(symbol, Cat) or symbol.name != node.cat:
        return None
    if not _demands_met(symbol.features, node.features()):
        return None
    if _meaning_carries(symbol.features, node.sem):
        return None
    bindings: Bindings = dict(item.bindings)
    bound = unify(ground(symbol.features, bindings), node.features(), bindings)
    if bound is None:
        return None
    return _Item(item.prod, item.dot + 1, item.start, _freeze(bound), item.children + (node,), item.weight + node.weight)


def _demands_met(wanted: Mapping[str, Any], got: Mapping[str, Any]) -> bool:
    """A literal feature demand must be *present* on the daughter, not merely unrefuted.

    Unification alone treats an absent feature as compatible, which is right for an
    agreement variable and wrong for subcategorisation: ``V[ditrans=true]`` would then
    match every verb, and "make me a sandwich" parses as a double-object verb whose
    object is "me". Selectional demands are checked here instead.
    """
    for key, value in wanted.items():
        if isinstance(value, FVar):
            continue
        if value is ABSENT:  # ``VP[tense=!]``: a modal's complement is a bare infinitive
            if key in got:
                return False
            continue
        if key not in got or got[key] != value:
            return False
    return True


def _meaning_carries(wanted: Mapping[str, Any], sem: Any) -> bool:
    """Whether a forbidden feature sits in the daughter's *meaning*.

    Tense reaches the frame, not the node: a production lifts it from the verb into
    what the clause says, so a category's feature list never mentions it. A
    prohibition has to look where the feature actually is, or ``VP[tense=!]`` would
    read "should gave food" happily and only generation would know better.
    """
    feats = getattr(getattr(sem, "frame", sem), "features", None)
    if not feats:
        return False
    return any(value is ABSENT and key in feats for key, value in wanted.items())


# ------------------------------------------------------------------ the cover


@dataclass(frozen=True)
class Reading:
    """One way to read the whole utterance: constituents plus what was skipped."""

    meanings: tuple[Any, ...]
    nodes: tuple[Node, ...]
    skipped: tuple[tuple[int, str], ...]
    score: float

    @property
    def complete(self) -> bool:
        return not self.skipped

    @property
    def guessed(self) -> tuple[tuple[str, str], ...]:
        """Words the lexicon did not have, with the category each was guessed as."""
        return tuple(g for node in self.nodes for g in node.guesses())

    def confidence(self, tokens: int) -> Score:
        """Lower when more of the utterance rested on guessed words. Uncalibrated."""
        share = len(self.guessed) / max(1, tokens)
        return Score(round(max(0.0, 1.0 - share), 3), "uncalibrated")

    def describe(self) -> str:
        parts = [m.describe() if hasattr(m, "describe") else repr(m) for m in self.meanings]
        tail = f"  (skipped: {' '.join(w for _, w in self.skipped)})" if self.skipped else ""
        return "; ".join(parts) + tail


def cover(grammar: Grammar, tokens: Sequence[str], chart: Chart, *, beam: int = 4,
          starts: Sequence[str] | None = None, clause_cost: float = -0.3) -> list[Reading]:
    """k-best sequences of top-level constituents, skipped tokens priced by the background.

    ``clause_cost`` charges each top-level constituent, so one clause that spans the
    utterance beats two that merely add up to it — which is what makes "Anem said the
    field failed" one report rather than two independent assertions. Chaining requests
    ("make a folder then list it") still pays it once per clause and wins anyway,
    because there is no single-clause reading to compete with.
    """
    n = len(tokens)
    cats = tuple(starts) if starts is not None else grammar.start
    best: list[list[tuple[float, tuple[tuple[str, Any], ...]]]] = [[] for _ in range(n + 1)]
    best[n] = [(0.0, ())]
    for i in range(n - 1, -1, -1):
        options: list[tuple[float, tuple[tuple[str, Any], ...]]] = []
        skip = grammar.lexicon.bg(tokens[i])
        for score, tail in best[i + 1]:
            options.append((score + skip, (("skip", i),) + tail))
        for node in chart.starting(i, cats):
            if node.end > i:
                for score, tail in best[node.end]:
                    options.append((score + node.weight + clause_cost, (("node", node),) + tail))
        options.sort(key=lambda o: (-o[0], _steps_key(o[1])))
        best[i] = options[:beam]

    readings = []
    for score, steps in best[0]:
        nodes = tuple(s[1] for s in steps if s[0] == "node")
        skipped = tuple((s[1], tokens[s[1]]) for s in steps if s[0] == "skip")
        readings.append(Reading(tuple(node.sem for node in nodes), nodes, skipped, score))
    return _dedupe(readings)


def _steps_key(steps: tuple) -> tuple:
    """A stable order for equal-scoring covers, so a tie always resolves the same way.

    Serial numbers, not content: totally ordered, cheap, and assigned in the chart's
    (deterministic) creation order. Preferring the cover that filled more roles was
    tried instead and changed nothing on either benchmark, so the cheap rule stands.
    """
    return tuple((0, s[1]) if s[0] == "skip" else (1, s[1].serial) for s in steps)


def _dedupe(readings: Sequence[Reading]) -> list[Reading]:
    out: list[Reading] = []
    seen: set = set()
    for reading in sorted(readings, key=lambda r: (-r.score, tuple(n.serial for n in r.nodes))):
        key = (tuple(_key_of(m) for m in reading.meanings), reading.skipped)
        if key not in seen:
            seen.add(key)
            out.append(reading)
    return out


@dataclass
class Understanding:
    """The result of reading one utterance: readings best-first, and what it cost."""

    text: str
    tokens: tuple[str, ...]
    readings: tuple[Reading, ...]
    ms: float = 0.0
    chart: Chart | None = field(default=None, repr=False)

    @property
    def best(self) -> Reading | None:
        return self.readings[0] if self.readings else None

    @property
    def ambiguous(self) -> bool:
        """Two readings of the *same* score: a genuine ambiguity, not just a ranking."""
        return len(self.readings) > 1 and abs(self.readings[0].score - self.readings[1].score) < 1e-9

    @property
    def meanings(self) -> tuple[Any, ...]:
        return self.best.meanings if self.best else ()

    @property
    def guessed(self) -> tuple[tuple[str, str], ...]:
        """Words read by guess rather than from the lexicon, for the caller to see."""
        return self.best.guessed if self.best else ()

    @property
    def confidence(self) -> Score:
        return self.best.confidence(len(self.tokens)) if self.best else Score(0.0, "uncalibrated")

    @property
    def skipped(self) -> tuple[str, ...]:
        return tuple(w for _, w in self.best.skipped) if self.best else self.tokens

    @property
    def coverage(self) -> float:
        return 1.0 if not self.tokens else 1.0 - len(self.skipped) / len(self.tokens)

    def describe(self) -> str:
        return self.best.describe() if self.best else "(no reading)"


def understand(grammar: Grammar, text: str, *, beam: int = 4, cap: int = 8,
               starts: Sequence[str] | None = None, clause_cost: float = -0.3) -> Understanding:
    """Parse one utterance: chart, then cover, then readings best-first."""
    t0 = time.perf_counter()
    tokens = tuple(tokenize(text))
    chart = build_chart(grammar, tokens, cap=cap)
    readings = cover(grammar, tokens, chart, beam=beam, starts=starts, clause_cost=clause_cost)
    return Understanding(text, tokens, tuple(readings), (time.perf_counter() - t0) * 1000, chart)
