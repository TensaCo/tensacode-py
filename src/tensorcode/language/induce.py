"""Learning a grammar from positive examples: hierarchical structure, not a finite-state table.

The core grammar in :mod:`tensorcode.language.english` is a seed. This module is how
structure the seed does not have gets *acquired*, and it is held to the test a
finite-state learner cannot pass: languages whose strings nest (brackets inside
brackets, ``a^n c b^n``), learned from short examples and checked on longer ones.

**The learner** is distributional learning of substitutable context-free languages
(Clark & Eyraud, *Polynomial identification in the limit of substitutable context-free
languages*, JMLR 2007). Two substrings that occur in the same context — the same
left part and right part of some example — are taken to be substitutable, and the
classes of substitutable substrings become nonterminals. Every way a substring splits
into two gives a binary rule between classes. For the substitutable languages this
identifies the language in the limit from positive data alone, in polynomial time.

It has a known limit, stated rather than hidden: ``a^n b^n`` is *not* substitutable
("a" and "aab" share the context (_, "b"), but "aabb" and "aababb" disagree), so the
learner over-generalises there. The known remedies are k,l-substitutability (Yoshinaka
2008, nonterminals indexed by the k symbols before and l after them) and learning the
syntactic concept lattice with membership queries (Clark 2010), where the agent may ask
whether a string is acceptable. Neither is implemented yet.

**The control** is a strictly k-local learner (accept a string iff every window of k
symbols in it, with boundary markers, was seen in training): the best finite-state
learner of its kind from positive data. On nested structure it must fail once
strings are longer than anything it saw.

Nothing here uses a neural network or a regular expression; strings are tuples of
symbols, so the same code learns over characters, words or word categories.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Hashable, Iterable, Sequence

Symbol = Hashable
String = tuple[Symbol, ...]


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[String, String] = {}

    def find(self, x: String) -> String:
        parent = self.parent.setdefault(x, x)
        if parent != x:
            parent = self.parent[x] = self.find(parent)
        return parent

    def union(self, a: String, b: String) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # the shorter (then lexically smaller) substring names the class: stable output
            keep, drop = sorted((ra, rb), key=lambda s: (len(s), repr(s)))
            self.parent[drop] = keep


@dataclass(frozen=True)
class CFG:
    """A context-free grammar in binary normal form over learned nonterminals."""

    start: int
    binary: frozenset[tuple[int, int, int]]         # (lhs, left, right)
    lexical: frozenset[tuple[int, Symbol]]          # (lhs, terminal)
    names: tuple[String, ...] = field(default=())  # a representative substring per nonterminal

    @property
    def nonterminals(self) -> int:
        return len(self.names)

    def size(self) -> int:
        return len(self.binary) + len(self.lexical)

    def accepts(self, string: Sequence[Symbol]) -> bool:
        """CYK recognition."""
        n = len(string)
        if n == 0:
            return False
        by_terminal: dict[Symbol, set[int]] = defaultdict(set)
        for lhs, a in self.lexical:
            by_terminal[a].add(lhs)
        by_pair: dict[tuple[int, int], set[int]] = defaultdict(set)
        for lhs, b, c in self.binary:
            by_pair[(b, c)].add(lhs)
        table: list[list[set[int]]] = [[set() for _ in range(n + 1)] for _ in range(n)]
        for i, a in enumerate(string):
            table[i][i + 1] = set(by_terminal.get(a, ()))
        for width in range(2, n + 1):
            for i in range(0, n - width + 1):
                j = i + width
                cell = table[i][j]
                for k in range(i + 1, j):
                    left, right = table[i][k], table[k][j]
                    if not left or not right:
                        continue
                    for b in left:
                        for c in right:
                            got = by_pair.get((b, c))
                            if got:
                                cell |= got
        return self.start in table[0][n]

    def productions(self, prefix: str = "L") -> list[str]:
        """The grammar as ``A -> B C`` / ``A -> "a"`` lines, for :func:`tensorcode.language.production`."""
        lines = [f'{prefix}{lhs} -> {prefix}{b} {prefix}{c}' for lhs, b, c in sorted(self.binary)]
        lines += [f'{prefix}{lhs} -> "{a}"' for lhs, a in sorted(self.lexical, key=repr)]
        return lines


def _substrings(sample: Iterable[String]) -> dict[String, set[tuple[String, String]]]:
    contexts: dict[String, set[tuple[String, String]]] = defaultdict(set)
    for s in sample:
        n = len(s)
        for i in range(n):
            for j in range(i + 1, n + 1):
                contexts[s[i:j]].add((s[:i], s[j:]))
    return contexts


def _grammar(sample: list[String], uf: _UnionFind, subs: Iterable[String]) -> CFG:
    ids: dict[String, int] = {}

    def nt(u: String) -> int:
        root = uf.find(u)
        if root not in ids:
            ids[root] = len(ids)
        return ids[root]

    binary, lexical = set(), set()
    for w in subs:
        if len(w) == 1:
            lexical.add((nt(w), w[0]))
            continue
        for k in range(1, len(w)):
            binary.add((nt(w), nt(w[:k]), nt(w[k:])))
    start = nt(sample[0])
    names = [()] * len(ids)
    for root, i in ids.items():
        names[i] = root
    return CFG(start, frozenset(binary), frozenset(lexical), tuple(names))


def learn_substitutable(sample: Iterable[Sequence[Symbol]]) -> CFG:
    """Clark & Eyraud's SGL: substrings sharing any context are one nonterminal."""
    strings = sorted({tuple(s) for s in sample if len(s)}, key=lambda s: (len(s), repr(s)))
    if not strings:
        raise ValueError("need at least one non-empty example")
    contexts = _substrings(strings)
    uf = _UnionFind()
    first_with: dict[tuple[String, String], String] = {}
    for u, cs in contexts.items():
        uf.find(u)
        for c in cs:
            if c in first_with:
                uf.union(first_with[c], u)
            else:
                first_with[c] = u
    return _grammar(strings, uf, contexts)


def learn_congruential(sample: Iterable[Sequence[Symbol]], *, complete_up_to: int | None = None) -> CFG:
    """Distributional learning with indirect negative evidence.

    Like :func:`learn_substitutable`, candidates for one nonterminal are substrings that
    share a context. Unlike it, a candidate merge of ``u`` and ``v`` is *refuted* when some
    context ``(l, r)`` of ``u`` gives a string ``l v r`` short enough that it should have
    been observed if it were in the language, and it was not (and the same with ``u`` and
    ``v`` swapped). ``)(`` and ``()`` share contexts in Dyck-1, but ``()`` stands alone
    and ``)(`` never does, which refutes that merge.

    **The assumption, stated:** ``complete_up_to`` is a length up to which the sample
    contains *every* string of the language (a length-complete presentation). Absence
    below it is then evidence. It defaults to the longest sample string, which is right
    only when the sample was built that way; for text that is not complete, this is the
    wrong learner (the expected-count version it needs does not exist yet).

    For languages whose syntactic congruence classes are witnessed within the bound,
    the classes found are those congruence classes, and the grammar built from them
    (every split of every substring, between classes) generates the language.
    """
    strings = sorted({tuple(s) for s in sample if len(s)}, key=lambda s: (len(s), repr(s)))
    if not strings:
        raise ValueError("need at least one non-empty example")
    bound = complete_up_to if complete_up_to is not None else max(len(s) for s in strings)
    observed = set(strings)
    contexts = _substrings(strings)

    def fits(u: String, v: String) -> bool:
        """Every context of u that would make a short-enough string, holds v too."""
        for left, right in contexts[u]:
            if len(left) + len(v) + len(right) <= bound and left + v + right not in observed:
                return False
        return True

    by_context: dict[tuple[String, String], list[String]] = defaultdict(list)
    for u, cs in contexts.items():
        for c in cs:
            by_context[c].append(u)
    uf = _UnionFind()
    for u in contexts:
        uf.find(u)
    members: dict[String, list[String]] = {u: [u] for u in contexts}
    for c in sorted(by_context, key=lambda c: (len(c[0]) + len(c[1]), repr(c))):
        group = sorted(by_context[c], key=lambda s: (len(s), repr(s)))
        for u in group[1:]:
            ru = uf.find(u)
            for v in group:
                if v is u:
                    break
                rv = uf.find(v)
                if ru == rv:
                    break
                # the whole classes must agree, not just this pair: otherwise a chain of
                # individually plausible merges joins two things the data separates
                if all(fits(a, b) and fits(b, a) for a in members[ru] for b in members[rv]):
                    uf.union(ru, rv)
                    keep = uf.find(ru)
                    drop = rv if keep == ru else ru
                    members[keep] = members[keep] + members.pop(drop)
                    break
    return _grammar(strings, uf, contexts)


@dataclass(frozen=True)
class KLocal:
    """The finite-state control: accept iff every k-window (with boundaries) was seen."""

    k: int
    windows: frozenset[tuple]

    def accepts(self, string: Sequence[Symbol]) -> bool:
        padded = ("<",) * (self.k - 1) + tuple(string) + (">",) * (self.k - 1)
        return all(padded[i:i + self.k] in self.windows for i in range(len(padded) - self.k + 1))


def learn_k_local(sample: Iterable[Sequence[Symbol]], k: int) -> KLocal:
    windows = set()
    for s in sample:
        padded = ("<",) * (k - 1) + tuple(s) + (">",) * (k - 1)
        windows.update(padded[i:i + k] for i in range(len(padded) - k + 1))
    return KLocal(k, frozenset(windows))
