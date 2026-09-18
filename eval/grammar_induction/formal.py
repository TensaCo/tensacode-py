"""Can the grammar learner acquire non-regular structure from positive examples?

    python -m eval.grammar_induction.formal [--out eval/results/grammar_induction_formal.json]

**Pre-registered, written before the first run (2026-09-18).**

Languages (membership is the mathematical definition below, not anything the learner uses):

=========  ================================  ================================================
name       language                          prediction for the substitutable learner (SGL)
=========  ================================  ================================================
dyck1      balanced ( )                      learned: accuracy >= 0.95 on longer strings
dyck2      balanced ( ) and [ ], nested      learned: accuracy >= 0.95
ancbn      a^n c b^n                         learned: accuracy >= 0.95
wcwr       w c reverse(w), w over {a, b}     learned: accuracy >= 0.95
anbn       a^n b^n, n >= 1                   NOT learned: not substitutable, so SGL
                                             over-generalises; accuracy well below 0.95
=========  ================================  ================================================

The control is the strictly k-local learner. **It is given its best k on the test set**
(k in 2..5, chosen after seeing test accuracy), which only makes it look better. The
prediction is that it stays well below SGL on the four substitutable languages, because
no finite window can check nesting depth it never saw.

Protocol: train on every string of the language up to a length bound; test on 200
positive strings strictly longer than any training string (up to 3x the bound) and 200
negatives of matched length — half near-miss edits of a positive (one symbol inserted,
deleted or changed, then confirmed outside the language), half random strings over
the same alphabet confirmed outside it. Seed 0. Accuracy with 95% Wilson intervals.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path[:0] = [str(Path(__file__).parents[2] / "src")]

from tensorcode.language.induce import learn_k_local, learn_substitutable  # noqa: E402

OUT = Path(__file__).parents[1] / "results" / "grammar_induction_formal.json"


def dyck(s: str, pairs: dict[str, str]) -> bool:
    stack = []
    closers = {v: k for k, v in pairs.items()}
    for ch in s:
        if ch in pairs:
            stack.append(ch)
        elif ch in closers:
            if not stack or stack.pop() != closers[ch]:
                return False
        else:
            return False
    return not stack and len(s) > 0


def ancbn(s: str) -> bool:
    n = s.find("c")
    return n >= 0 and s.count("c") == 1 and s[:n] == "a" * n and s[n + 1:] == "b" * n


def wcwr(s: str) -> bool:
    if s.count("c") != 1:
        return False
    w, rest = s.split("c")
    return set(w) <= {"a", "b"} and rest == w[::-1]


def anbn(s: str) -> bool:
    n = len(s) // 2
    return len(s) % 2 == 0 and n >= 1 and s == "a" * n + "b" * n


def gen_dyck(rng: random.Random, length: int, pairs: dict[str, str]) -> str:
    """A uniform-ish random balanced string of exactly ``length`` (even) symbols."""
    opens = list(pairs)
    out, stack, remaining = [], [], length
    while remaining:
        can_open = remaining - len(stack) >= 2
        if stack and (not can_open or rng.random() < 0.5):
            out.append(pairs[stack.pop()])
        else:
            o = rng.choice(opens)
            stack.append(o)
            out.append(o)
        remaining -= 1
    return "".join(out)


LANGUAGES = {
    "dyck1": dict(alphabet="()", member=lambda s: dyck(s, {"(": ")"}), train_len=10,
                  gen=lambda rng, n: gen_dyck(rng, n - n % 2, {"(": ")"})),
    "dyck2": dict(alphabet="()[]", member=lambda s: dyck(s, {"(": ")", "[": "]"}), train_len=8,
                  gen=lambda rng, n: gen_dyck(rng, n - n % 2, {"(": ")", "[": "]"})),
    "ancbn": dict(alphabet="acb", member=ancbn, train_len=9,
                  gen=lambda rng, n: "a" * ((n - 1) // 2) + "c" + "b" * ((n - 1) // 2)),
    "wcwr": dict(alphabet="acb", member=wcwr, train_len=9,
                 gen=lambda rng, n: (lambda w: w + "c" + w[::-1])("".join(rng.choice("ab") for _ in range((n - 1) // 2)))),
    "anbn": dict(alphabet="ab", member=anbn, train_len=10,
                 gen=lambda rng, n: "a" * (n // 2) + "b" * (n // 2)),
}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 3), round(c + h, 3))


def training_set(spec: dict) -> list[str]:
    out = []
    for n in range(1, spec["train_len"] + 1):
        for t in itertools.product(spec["alphabet"], repeat=n):
            s = "".join(t)
            if spec["member"](s):
                out.append(s)
    return out


def test_set(spec: dict, rng: random.Random, n: int = 200) -> tuple[list[str], list[str]]:
    lo, hi = spec["train_len"] + 1, 3 * spec["train_len"]
    member, alphabet = spec["member"], spec["alphabet"]
    pos: list[str] = []
    tries = 0
    while len(pos) < n and tries < 100000:
        tries += 1
        s = spec["gen"](rng, rng.randint(lo, hi))
        if len(s) >= lo and member(s):
            pos.append(s)
    neg: list[str] = []
    tries = 0
    while len(neg) < n // 2 and tries < 100000:  # near misses
        tries += 1
        s = list(rng.choice(pos))
        i = rng.randrange(len(s))
        op = rng.choice(("ins", "del", "sub"))
        if op == "ins":
            s.insert(i, rng.choice(alphabet))
        elif op == "del":
            del s[i]
        else:
            s[i] = rng.choice([a for a in alphabet if a != s[i]])
        t = "".join(s)
        if len(t) >= lo and not member(t):
            neg.append(t)
    tries = 0
    while len(neg) < n and tries < 100000:  # random strings of matched length
        tries += 1
        t = "".join(rng.choice(alphabet) for _ in range(rng.randint(lo, hi)))
        if not member(t):
            neg.append(t)
    return pos, neg


def score(accepts, pos: list[str], neg: list[str]) -> dict:
    tp = sum(1 for s in pos if accepts(tuple(s)))
    tn = sum(1 for s in neg if not accepts(tuple(s)))
    n = len(pos) + len(neg)
    return {"accuracy": round((tp + tn) / n, 3), "ci95": wilson(tp + tn, n),
            "accept_positive": f"{tp}/{len(pos)}", "reject_negative": f"{tn}/{len(neg)}"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    rng = random.Random(0)
    rows = {}
    for name, spec in LANGUAGES.items():
        train = training_set(spec)
        pos, neg = test_set(spec, rng)
        t0 = time.perf_counter()
        g = learn_substitutable(train)
        learn_s = time.perf_counter() - t0
        sgl = score(g.accepts, pos, neg)
        controls = {k: score(learn_k_local(train, k).accepts, pos, neg) for k in (2, 3, 4, 5)}
        best_k = max(controls, key=lambda k: controls[k]["accuracy"])
        rows[name] = {
            "train": {"strings": len(train), "max_len": spec["train_len"]},
            "test": {"positives": len(pos), "negatives": len(neg), "lengths": [spec["train_len"] + 1, 3 * spec["train_len"]]},
            "sgl": {**sgl, "nonterminals": g.nonterminals, "rules": g.size(), "learn_seconds": round(learn_s, 3)},
            "k_local_best": {"k": best_k, **controls[best_k]},
            "k_local_all": controls,
        }
        print(f"{name:6} SGL {sgl['accuracy']:.3f} {sgl['ci95']}  ({g.nonterminals} NT, {g.size()} rules) | "
              f"k-local best k={best_k} {controls[best_k]['accuracy']:.3f} {controls[best_k]['ci95']}")
    result = {
        "eval": "grammar_induction_formal",
        "date": "2026-09-18",
        "preregistered": "see module docstring (written before the first run)",
        "seed": 0,
        "provenance": {"languages": "mathematical definitions in this file", "grader": "exact membership functions",
                       "author_of_learner": "same author; the languages are standard textbook examples, not derived from the learner"},
        "languages": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
