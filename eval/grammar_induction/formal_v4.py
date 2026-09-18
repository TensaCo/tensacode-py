"""Grammar induction v4: the testability horizon, re-measured where it was not developed.

    python -m eval.grammar_induction.formal_v4

**Why v4 exists.** v3 found the congruential learner exact on Dyck-1, Dyck-2, a^n c b^n
and w c w^R, and wrong on a^n b^n (precision 0.02): ``aaaaabbbb`` (length 9 of a bound
of 10) could only be tested in contexts short enough to fit, so nothing refuted merging
it with ``a``. The fix — only substrings up to half the bound become nonterminals — was
written *after* seeing that failure, so by the project's rules it counts only if it
also holds where it was not developed.

**Pre-registered, written before the first v4 run (2026-09-18).**

Two groups, same measures as v2/v3 (recall on longer members, precision by sampling the
learner's own language, exhaustive errors where the alphabet has <= 3 symbols):

* ``dev`` — the five v3 languages, which the fix was developed against;
* ``fresh`` — languages the learner and the fix never saw, and a different bound:

  ===========  ===================================================  =========
  name         language                                              bound
  ===========  ===================================================  =========
  anb2n        a^n b^(2n), n >= 1                                    12
  dyck3        balanced ( ) [ ] { }                                  8
  anbmcmdn     a^n b^m c^m d^n, n, m >= 1 (nested counts)            10
  pal3         w c reverse(w), w over {a, b, d}                      9
  ===========  ===================================================  =========

Predictions for ``congruential`` (with the horizon):

* dev: all five exact — precision and recall >= 0.95, zero exhaustive errors;
* fresh: at least 3 of the 4 with precision and recall >= 0.95. (a^n b^(2n) has the same
  counting structure as a^n b^n, so it is the likeliest to fail if the fix only
  patched one case.)

If fewer than 3 fresh languages meet the bar, the horizon is recorded as a fix for
one language, not as a property of the learner.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path[:0] = [str(Path(__file__).parents[2] / "src"), str(Path(__file__).parents[2])]

from eval.grammar_induction.formal import LANGUAGES, dyck, gen_dyck, test_set, training_set  # noqa: E402
from eval.grammar_induction.formal_v2 import measure, sample_cfg  # noqa: E402
from tensorcode.language.induce import learn_congruential  # noqa: E402

OUT = Path(__file__).parents[1] / "results" / "grammar_induction_formal_v4.json"


def anb2n(s: str) -> bool:
    n = s.count("a")
    return n >= 1 and s == "a" * n + "b" * (2 * n)


def anbmcmdn(s: str) -> bool:
    import itertools as it

    groups = [(k, len(list(g))) for k, g in it.groupby(s)]
    return [k for k, _ in groups] == ["a", "b", "c", "d"] and groups[0][1] == groups[3][1] and groups[1][1] == groups[2][1]


def pal3(s: str) -> bool:
    if s.count("c") != 1:
        return False
    w, rest = s.split("c")
    return set(w) <= {"a", "b", "d"} and rest == w[::-1]


FRESH = {
    "anb2n": dict(alphabet="ab", member=anb2n, train_len=12,
                  gen=lambda rng, n: "a" * max(1, n // 3) + "b" * (2 * max(1, n // 3))),
    "dyck3": dict(alphabet="()[]{}", member=lambda s: dyck(s, {"(": ")", "[": "]", "{": "}"}), train_len=8,
                  gen=lambda rng, n: gen_dyck(rng, n - n % 2, {"(": ")", "[": "]", "{": "}"})),
    "anbmcmdn": dict(alphabet="abcd", member=anbmcmdn, train_len=10,
                     gen=lambda rng, n: (lambda a, b: "a" * a + "b" * b + "c" * b + "d" * a)(*(lambda a: (a, max(1, (n - 2 * a) // 2)))(rng.randint(1, max(1, n // 2 - 1))))),
    "pal3": dict(alphabet="abdc", member=pal3, train_len=9,
                 gen=lambda rng, n: (lambda w: w + "c" + w[::-1])("".join(rng.choice("abd") for _ in range((n - 1) // 2)))),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    rng = random.Random(0)
    rows = {}
    for group, langs in (("dev", LANGUAGES), ("fresh", FRESH)):
        for name, spec in langs.items():
            train = training_set(spec)
            pos, _ = test_set(spec, rng)
            lo, hi = spec["train_len"] + 1, 3 * spec["train_len"]
            t0 = time.perf_counter()
            g = learn_congruential(train, complete_up_to=spec["train_len"])
            r = measure(name, spec, g.accepts, sample_cfg(g, rng, lo, hi, 200), pos)
            r.update(train_strings=len(train), nonterminals=g.nonterminals, rules=g.size(), seconds=round(time.perf_counter() - t0, 2))
            ok = (r["precision"] or 0) >= 0.95 and r["recall"] >= 0.95 and (not r["exhaustive"] or r["exhaustive"]["false_accepts"] + r["exhaustive"]["false_rejects"] == 0)
            r["meets_bar"] = ok
            rows[f"{group}/{name}"] = r
            ex = r["exhaustive"]
            print(f"{group:5} {name:9} P={r['precision']} R={r['recall']}"
                  + (f" exhaustive FA={ex['false_accepts']} FR={ex['false_rejects']}/{ex['strings']}" if ex else "")
                  + f" ({g.nonterminals} NT, {r['seconds']}s) {'MEETS' if ok else 'FAILS'}", flush=True)
    fresh_ok = sum(1 for k, v in rows.items() if k.startswith("fresh/") and v["meets_bar"])
    dev_ok = sum(1 for k, v in rows.items() if k.startswith("dev/") and v["meets_bar"])
    verdict = {"dev_meeting_bar": f"{dev_ok}/5", "fresh_meeting_bar": f"{fresh_ok}/4",
               "prediction_dev": "5/5", "prediction_fresh": ">= 3/4",
               "held": dev_ok == 5 and fresh_ok >= 3}
    print(verdict)
    args.out.write_text(json.dumps({"eval": "grammar_induction_formal_v4", "date": "2026-09-18", "seed": 0,
                                    "preregistered": "module docstring, committed before the first run",
                                    "verdict": verdict, "languages": rows}, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
