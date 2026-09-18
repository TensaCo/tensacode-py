"""Grammar induction on non-regular languages, v2: precision is measured, not assumed.

    python -m eval.grammar_induction.formal_v2

**Why v2 exists.** v1 (``formal.py``, results kept unmodified in
``eval/results/grammar_induction_formal.json``) scored the substitutable learner at
0.98–1.00 on all five languages, including ``a^n b^n``, which it was predicted to fail.
Probing showed v1's negatives were too easy: near-miss edits and random strings almost
never hit what the learner over-generalises to. The learned Dyck-1 grammar accepts
``)(`` and disagrees with the language on 1,078 strings up to length 12; the ``a^n b^n``
grammar accepts ``aababb``. v1 measured recall and called it accuracy.

**Pre-registered, written before the first v2 run (2026-09-18).**

Measures, on lengths strictly longer than every training string (train bound L, test L+1..3L):

* **recall**: fraction of 200 random members the learner accepts;
* **precision**: fraction of 200 strings *sampled uniformly from the learner's own
  language* (by length, then uniformly among its accepted strings of that length) that
  are members — the direct measure of over-generalisation;
* **exhaustive**: for alphabets of size <= 3, every string of length L+1..L+4, counting
  false accepts and false rejects.

Predictions for SGL (the substitutable learner):

=========  ==========================================================================
dyck1      over-generalises (not substitutable: ``)(`` ~ ``()``): precision < 0.95
dyck2      over-generalises for the same reason: precision < 0.95
ancbn      learned exactly: precision and recall >= 0.95, zero exhaustive errors
wcwr       learned exactly: precision and recall >= 0.95, zero exhaustive errors
anbn       over-generalises (not substitutable): precision < 0.95
=========  ==========================================================================

The k-local control (best k on the test, which flatters it) is predicted to have
precision < 0.95 on all five, since no finite window enforces unbounded nesting.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
import time
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

sys.path[:0] = [str(Path(__file__).parents[2] / "src"), str(Path(__file__).parents[2])]

from eval.grammar_induction.formal import LANGUAGES, test_set, training_set, wilson  # noqa: E402
from tensorcode.language.induce import CFG, KLocal, learn_k_local, learn_substitutable  # noqa: E402

OUT = Path(__file__).parents[1] / "results" / "grammar_induction_formal_v2.json"


# ------------------------------------------------------------ sampling a learner's language

class CFGSampler:
    """Uniform sampling of strings of a given length from a binary-normal-form grammar.

    Counts derivations per (nonterminal, length); an ambiguous grammar over-weights
    strings with more derivations, so a sampled string is then accepted-checked and
    the estimate is of *derivations*, which is what the grammar puts weight on.
    """

    def __init__(self, g: CFG, max_len: int) -> None:
        self.g = g
        self.by_lhs = defaultdict(list)
        for lhs, b, c in g.binary:
            self.by_lhs[lhs].append((b, c))
        self.lex = defaultdict(list)
        for lhs, a in g.lexical:
            self.lex[lhs].append(a)
        self.max_len = max_len

    @lru_cache(maxsize=None)
    def count(self, nt: int, n: int) -> int:
        if n == 1:
            return len(self.lex[nt])
        return sum(self.count(b, k) * self.count(c, n - k) for b, c in self.by_lhs[nt] for k in range(1, n))

    def sample(self, rng: random.Random, nt: int, n: int) -> list:
        if n == 1:
            return [rng.choice(self.lex[nt])]
        total = self.count(nt, n)
        r = rng.randrange(total)
        for b, c in self.by_lhs[nt]:
            for k in range(1, n):
                w = self.count(b, k) * self.count(c, n - k)
                if r < w:
                    return self.sample(rng, b, k) + self.sample(rng, c, n - k)
                r -= w
        raise AssertionError("unreachable")


def sample_cfg(g: CFG, rng: random.Random, lo: int, hi: int, n: int) -> list[str]:
    s = CFGSampler(g, hi)
    lengths = [m for m in range(lo, hi + 1) if s.count(g.start, m) > 0]
    if not lengths:
        return []
    return ["".join(s.sample(rng, g.start, rng.choice(lengths))) for _ in range(n)]


def sample_klocal(m: KLocal, alphabet: str, rng: random.Random, lo: int, hi: int, n: int) -> list[str]:
    """Uniform over accepted strings of a random accepted length, by path counting."""
    k = m.k
    start = ("<",) * (k - 1)

    @lru_cache(maxsize=None)
    def count(state: tuple, remaining: int) -> int:
        if remaining == 0:
            tail = state + (">",) * (k - 1)
            return int(all(tail[i:i + k] in m.windows for i in range(len(tail) - k + 1)))
        return sum(count((state + (a,))[1:], remaining - 1) for a in alphabet if state + (a,) in m.windows)

    lengths = [L for L in range(lo, hi + 1) if count(start, L) > 0]
    out = []
    for _ in range(n if lengths else 0):
        L, state, s = rng.choice(lengths), start, []
        for remaining in range(L, 0, -1):
            options = [(a, count((state + (a,))[1:], remaining - 1)) for a in alphabet if state + (a,) in m.windows]
            r = rng.randrange(sum(w for _, w in options))
            for a, w in options:
                if r < w:
                    s.append(a)
                    state = (state + (a,))[1:]
                    break
                r -= w
        out.append("".join(s))
    return out


def exhaustive(accepts, member, alphabet: str, lo: int, hi: int) -> dict:
    fa = fr = total = 0
    examples = []
    for n in range(lo, hi + 1):
        for t in itertools.product(alphabet, repeat=n):
            s = "".join(t)
            a, m = accepts(t), member(s)
            total += 1
            if a and not m:
                fa += 1
                if len(examples) < 3:
                    examples.append(s)
            elif m and not a:
                fr += 1
    return {"strings": total, "false_accepts": fa, "false_rejects": fr, "false_accept_examples": examples}


def measure(name: str, spec: dict, accepts, sampled: list[str], pos: list[str]) -> dict:
    member = spec["member"]
    rec = sum(1 for s in pos if accepts(tuple(s)))
    prec = sum(1 for s in sampled if member(s))
    lo, hi = spec["train_len"] + 1, spec["train_len"] + 4
    ex = exhaustive(accepts, member, spec["alphabet"], lo, hi) if len(spec["alphabet"]) <= 3 else None
    return {"recall": round(rec / len(pos), 3), "recall_ci95": wilson(rec, len(pos)),
            "precision": round(prec / len(sampled), 3) if sampled else None,
            "precision_ci95": wilson(prec, len(sampled)) if sampled else None,
            "exhaustive": ex}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    rng = random.Random(0)
    rows = {}
    for name, spec in LANGUAGES.items():
        train = training_set(spec)
        pos, _ = test_set(spec, rng)
        lo, hi = spec["train_len"] + 1, 3 * spec["train_len"]
        t0 = time.perf_counter()
        g = learn_substitutable(train)
        sgl = measure(name, spec, g.accepts, sample_cfg(g, rng, lo, hi, 200), pos)
        sgl.update(nonterminals=g.nonterminals, rules=g.size(), seconds=round(time.perf_counter() - t0, 2))
        controls = {}
        for k in (2, 3, 4, 5):
            m = learn_k_local(train, k)
            controls[k] = measure(name, spec, m.accepts, sample_klocal(m, spec["alphabet"], rng, lo, hi, 200), pos)
        def f1(r):
            p, c = r["precision"] or 0.0, r["recall"]
            return 0.0 if p + c == 0 else 2 * p * c / (p + c)
        best = max(controls, key=lambda k: f1(controls[k]))
        rows[name] = {"train_strings": len(train), "train_max_len": spec["train_len"], "test_lengths": [lo, hi],
                      "sgl": sgl, "k_local_best": {"k": best, **controls[best]}}
        ex = sgl["exhaustive"]
        print(f"{name:6} SGL P={sgl['precision']} R={sgl['recall']}"
              + (f" exhaustive FA={ex['false_accepts']}/{ex['strings']} e.g. {ex['false_accept_examples']}" if ex else "")
              + f" | k-local k={best} P={controls[best]['precision']} R={controls[best]['recall']}")
    args.out.write_text(json.dumps({"eval": "grammar_induction_formal_v2", "date": "2026-09-18", "seed": 0,
                                    "preregistered": "module docstring, committed before the first run",
                                    "supersedes": "grammar_induction_formal.json (v1 measured recall only)",
                                    "languages": rows}, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
