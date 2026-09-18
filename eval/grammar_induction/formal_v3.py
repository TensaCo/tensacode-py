"""Grammar induction v3: the congruential learner (indirect negative evidence) against v2's.

    python -m eval.grammar_induction.formal_v3

Same measures as v2 (recall on longer members, precision by sampling the learner's own
language, exhaustive errors for small alphabets), same languages, same seed.

**Pre-registered, written before the first v3 run (2026-09-18).**

Arms:

* ``congruential`` — :func:`learn_congruential`, trained on the length-complete sample
  (every member up to the bound), which is the presentation its assumption needs.
* ``congruential_half`` — the same learner on a random half of those strings, so its
  assumption (absence below the bound is evidence) is false.
* ``sgl`` — v2's learner, repeated as the reference.

Predictions:

=====================  ==================================================================
congruential           all five languages (dyck1, dyck2, ancbn, wcwr, anbn): precision
                       and recall >= 0.95, and zero exhaustive errors where measured
congruential_half      fails (precision or recall < 0.95) on at least 3 of the 5: with
                       its assumption false it refutes true merges and under-generalises
sgl                    as in v2: exact on ancbn and wcwr, precision < 0.95 on the rest
=====================  ==================================================================
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path[:0] = [str(Path(__file__).parents[2] / "src"), str(Path(__file__).parents[2])]

from eval.grammar_induction.formal import LANGUAGES, test_set, training_set  # noqa: E402
from eval.grammar_induction.formal_v2 import measure, sample_cfg  # noqa: E402
from tensorcode.language.induce import learn_congruential, learn_substitutable  # noqa: E402

OUT = Path(__file__).parents[1] / "results" / "grammar_induction_formal_v3.json"


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
        half = random.Random(1).sample(train, len(train) // 2)
        arms = {}
        for arm, learn, data in (("congruential", lambda d: learn_congruential(d, complete_up_to=spec["train_len"]), train),
                                 ("congruential_half", lambda d: learn_congruential(d, complete_up_to=spec["train_len"]), half),
                                 ("sgl", learn_substitutable, train)):
            t0 = time.perf_counter()
            g = learn(data)
            r = measure(name, spec, g.accepts, sample_cfg(g, rng, lo, hi, 200), pos)
            r.update(train_strings=len(data), nonterminals=g.nonterminals, rules=g.size(), seconds=round(time.perf_counter() - t0, 2))
            arms[arm] = r
            ex = r["exhaustive"]
            print(f"{name:6} {arm:18} P={r['precision']} R={r['recall']}"
                  + (f" exhaustive FA={ex['false_accepts']} FR={ex['false_rejects']}/{ex['strings']}" if ex else "")
                  + f" ({g.nonterminals} NT, {r['seconds']}s)", flush=True)
        rows[name] = {"train_max_len": spec["train_len"], "test_lengths": [lo, hi], "arms": arms}
    args.out.write_text(json.dumps({"eval": "grammar_induction_formal_v3", "date": "2026-09-18", "seed": 0,
                                    "preregistered": "module docstring, committed before the first run",
                                    "languages": rows}, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
