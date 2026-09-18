"""Measure the symbolic parser against the assistant's regexes, and the induction controls.

    python eval/language_benchmark.py            # writes eval/results/language_benchmark.json

Three things are measured, because the interesting comparison is not one number:

1. **The assistant's own benchmark** (``tests/test_assistant_language.py``): 152
   utterances with expected ``(act, slots)``. The regex parser was written against
   these, so it is the ceiling here, not a rival.
2. **Compositional utterances** the act vocabulary cannot express — reported speech,
   negation, modality, quantifiers, comparatives, tense. This is what a grammar
   buys and a pattern list cannot.
3. **Induction with its controls**, including an artifact that reads vocabulary and
   is refused.

The compositional set is run **twice**: once with the content words in the lexicon,
and once against bare ``ENGLISH``, where every content word is unknown. The second is
the realistic case for the simulation — no fixed lexicon holds the words villagers
use — and the two numbers are reported separately because they are different claims.

No model is called. Everything here runs on the standard library.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from tensacode.language import ENGLISH, Context, Frame, Question, Request, realize, resolve, understand, words  # noqa: E402
from tensacode.language.domains.desktop import DESKTOP, read_request  # noqa: E402
from tensacode.learning import (  # noqa: E402
    candidate_literals, decision_list, verify_decision_list,
)

OUT = ROOT / "eval" / "results" / "language_benchmark.json"

# ------------------------------------------------------- 1. the act benchmark


def act_benchmark() -> dict:
    from tests.test_assistant_language import CASES

    from examples.browser_agents.assistant.language import parse_message

    grammar_ok, regex_ok = 0, 0
    times: list[float] = []
    coverage: list[float] = []
    ambiguous = 0
    failures: list[dict] = []
    for text, expected in CASES:
        start = time.perf_counter()
        got, understanding = read_request(text)
        times.append((time.perf_counter() - start) * 1000)
        coverage.append(understanding.coverage)
        ambiguous += bool(understanding.ambiguous)
        mine = [(a.act, dict(a.slots)) for a in got]
        good = len(mine) == len(expected) and all(
            act == want_act and all(slots.get(k) == v for k, v in want_slots.items())
            for (act, slots), (want_act, want_slots) in zip(mine, expected))
        grammar_ok += good
        if not good:
            failures.append({"text": text, "grammar": [f"{a}({s})" for a, s in mine],
                             "expected": [f"{a}({s})" for a, s in expected]})
        frames = [(f.act, f.slots) for f in parse_message(text)]
        regex_ok += len(frames) == len(expected) and all(
            act == want_act and all(slots.get(k) == v for k, v in want_slots.items())
            for (act, slots), (want_act, want_slots) in zip(frames, expected))
    times.sort()
    return {
        "cases": len(CASES),
        "grammar_correct": grammar_ok,
        "grammar_accuracy": round(grammar_ok / len(CASES), 4),
        "regex_correct": regex_ok,
        "regex_accuracy": round(regex_ok / len(CASES), 4),
        "ms_per_utterance": {"p50": round(times[len(times) // 2], 2), "p95": round(times[int(0.95 * len(times))], 2)},
        "mean_token_coverage": round(statistics.mean(coverage), 4),
        "utterances_with_equal_scoring_readings": ambiguous,
        "failures": failures,
    }


# -------------------------------------------- 2. compositional understanding

VILLAGE = ENGLISH.extend(entries=[
    *words("field", cat="N", sem="field"), *words("grain", cat="N", sem="grain"),
    *words("harvest", cat="N", sem="harvest"), *words("well", cat="N", sem="well"),
    *words("fail", cat="V", sem="fail"), *words("arrive", cat="V", sem="arrive"),
    *words("share", cat="V", sem="share"), *words("flood", cat="V", sem="flood"),
    *words("north", "south", cat="Adj"), *words("good", cat="Adj"),
    *words("better", cat="Adj", sem="good", degree="comparative"),
    *words("dry", cat="Adj", sem="dry"),
])

#: (utterance, predicate, checks on the frame). These are the constructions a
#: pattern list cannot represent, so the comparison is about structure, not slots.
COMPOSITIONAL = [
    ("the north field failed", "fail", {"tense": "past"}),
    ("the grain did not arrive", "arrive", {"polarity": "negative", "tense": "past"}),
    ("the grain has arrived", "arrive", {"aspect": "perfect"}),
    ("everyone must share grain", "share", {"modality": "must"}),
    ("no one shares grain", "share", {}),
    ("the north field is better than the south field", "good", {"degree": "comparative"}),
    ("Anem said the north field failed", "say", {"reported": "fail"}),
    ("Bera said the grain did not arrive", "say", {"reported": "arrive", "reported_negative": True}),
    ("Anem said that the well is dry", "say", {"reported": "dry"}),
    ("the well might flood", "flood", {"modality": "may"}),
    ("all the fields failed", "fail", {"quantifier": "all"}),
    ("some grain arrived", "arrive", {"quantifier": "some"}),
    ("share the grain", "!request", {"act": "share"}),
    ("did the north field fail", "?question", {"asked": "polarity"}),
    ("who shares grain", "?question", {"asked": "subject"}),
]


#: The open-vocabulary set, with the expectations open vocabulary can actually meet.
#: Two differences from the in-lexicon set, both deliberate:
#:
#: * the predicate of an unseen verb is its **stem after the suffix is removed**
#:   ("arriv" for "arrived"). That is an internal identifier, not a dictionary lemma,
#:   and it is stable in both directions, which is what a speaker and a hearer need;
#: * an irregular comparative ("better" for "good") and an unseen adjective cannot be
#:   resolved to a lemma with no lexicon, so those cases are checked on **structure** —
#:   that a comparison carries a standard, and that a report carries a nested clause.
OPEN_VOCABULARY = [
    ("the north field failed", "fail", {"tense": "past"}),
    ("the grain did not arrive", "arrive", {"polarity": "negative", "tense": "past"}),
    # these three read "arriv" until the stemmer was fixed: the expectation had the
    # defect written into it, which is what a benchmark built beside the system does
    ("the grain has arrived", "arrive", {"aspect": "perfect"}),
    ("everyone must share grain", "share", {"modality": "must"}),
    ("no one shares grain", "share", {"quantifier": "none"}),
    ("the north field is better than the south field", "*", {"has_role": "standard"}),
    ("Anem said the north field failed", "say", {"reported": "fail"}),
    ("Bera said the grain did not arrive", "say", {"reported": "arrive", "reported_negative": True}),
    ("Anem said that the well is dry", "say", {"reported": "*"}),
    ("the well might flood", "flood", {"modality": "may"}),
    ("all the fields failed", "fail", {"quantifier": "all"}),
    ("some grain arrived", "arrive", {"quantifier": "some"}),
    ("share the grain", "!request", {"act": "share"}),
    ("did the north field fail", "?question", {"asked": "polarity"}),
    ("who shares grain", "?question", {"asked": "subject"}),
    # the coordinator's own transcripts, which is where the gap showed up
    ("Anem said the north field did not fail", "say", {"reported": "fail", "reported_negative": True}),
    ("the north field did not fail", "fail", {"polarity": "negative", "tense": "past"}),
    ("Anem told me the grain arrived", "tell", {"reported": "arrive"}),
    ("the grain store is empty", "*", {"subject_text": "grain store"}),
]


def compositional(grammar=VILLAGE, cases=None, *, label="in_lexicon") -> dict:
    from examples.browser_agents.assistant.language import parse_message

    rows = []
    ok = trips = 0
    cases = cases if cases is not None else COMPOSITIONAL
    for text, predicate, checks in cases:
        got = understand(grammar, text)
        meaning = got.meanings[0] if got.meanings else None
        good, detail = _check(meaning, predicate, checks)
        ok += good
        said = realize(grammar, meaning) if meaning is not None else None
        round_trip = False
        if said:
            back = understand(grammar, said)
            round_trip = bool(back.meanings) and repr(back.meanings[0]) == repr(meaning)
        trips += round_trip
        regex = [f.act for f in parse_message(text)]
        rows.append({"text": text, "grammar_reading": got.describe()[:160], "correct": good, "detail": detail,
                     "said_again": said, "round_trip": round_trip, "regex_acts": regex,
                     "coverage": round(got.coverage, 3), "guessed": [list(g) for g in got.guessed],
                     "confidence": got.confidence.value})
    full = sum(r["coverage"] == 1.0 for r in rows)
    return {
        "vocabulary": label,
        "cases": len(cases),
        "grammar_correct": ok,
        "grammar_accuracy": round(ok / len(cases), 4),
        "fully_covered": full,
        "regex_unknown_or_wrong": sum(1 for r in rows if set(r["regex_acts"]) <= {"unknown"} or not r["regex_acts"]),
        "round_trips": trips,
        "round_trip_rate": round(trips / len(cases), 4),
        "rows": rows,
    }


def _check(meaning, predicate, checks) -> tuple[bool, str]:
    if meaning is None:
        return False, "no reading"
    if predicate == "!request":
        return (isinstance(meaning, Request) and meaning.act == checks["act"]), f"{type(meaning).__name__}"
    if predicate == "?question":
        return (isinstance(meaning, Question) and meaning.asked == checks["asked"]), f"{type(meaning).__name__}"
    if not isinstance(meaning, Frame):
        return False, f"not a frame: {type(meaning).__name__}"
    if predicate != "*" and meaning.predicate != predicate:
        return False, f"predicate {meaning.predicate!r}"
    for key, value in checks.items():
        if key == "has_role":
            if not any(f.role(value) is not None for f in meaning.walk()):
                return False, f"no {value} role"
        elif key == "subject_text":
            subject = meaning.role("subject")
            if subject is None or getattr(subject, "text", None) != value:
                return False, f"subject {getattr(subject, 'text', None)!r}"
        elif key == "reported":
            inner = meaning.role("content")
            if not isinstance(inner, Frame) or (value != "*" and inner.predicate != value):
                return False, "reported content missing"
        elif key == "reported_negative":
            inner = meaning.role("content")
            if not isinstance(inner, Frame) or inner.negated != value:
                return False, "reported polarity lost"
        elif key == "quantifier":
            if not any(e.features.get("quantifier") == value for e in meaning.entities()):
                return False, f"quantifier {value!r} missing"
        elif meaning.feature(key) != value:
            return False, f"{key}={meaning.feature(key)!r}"
    return True, "ok"


def discourse_demo() -> dict:
    """Pronouns resolved against the conversation, and refused when two antecedents tie."""
    context = Context()
    first = understand(VILLAGE, "the north field failed").meanings[0]
    context.observe(first)
    resolved = resolve(understand(VILLAGE, "it failed").meanings[0], context)
    one = resolved.role("subject").resolved

    context2 = Context()
    for text in ("the north field failed", "the south field failed"):
        context2.observe(understand(VILLAGE, text).meanings[0])
    tied = resolve(understand(VILLAGE, "it failed").meanings[0], context2)
    ambiguous = tied.role("subject").candidates
    return {"single_antecedent_resolved": bool(one),
            "tied_antecedents_left_unresolved": len(ambiguous) >= 2,
            "candidates": [c.text for c in ambiguous]}


# ------------------------------------------------- 3. induction and controls


def induction() -> dict:
    import random

    def cases(n, seed):
        rng = random.Random(seed)
        out = []
        for i in range(n):
            urgent, paid = rng.random() < 0.5, rng.random() < 0.5
            region = rng.choice(["north", "south", "east"])
            label = "fast" if (urgent and paid) else "queued" if urgent else "normal"
            out.append((frozenset({("urgent", urgent), ("paid", paid), ("region", region), ("index", i % 7)}), label))
        return out

    train, held = cases(120, 11), cases(60, 12)
    literals = candidate_literals(train)
    rules = decision_list(train, literals)
    # the wrong-question control keeps the same facts and the same label multiset, and
    # only breaks the pairing. A constant label would not be the same question asked
    # wrongly — it would be a different label distribution, and any artifact scores
    # oddly against that.
    labels = [y for _, y in held]
    rotated = labels[7:] + labels[:7]
    real = verify_decision_list(rules, train=train, held_out=held, literals=literals,
                                shifted=[(f, y) for (f, _), y in zip(held, rotated)])

    # an artifact that can only be reading vocabulary: the label *is* a symbol
    vocab_cases = [(frozenset({("region", r), ("urgent", True)}), r) for r in ["north", "south", "east", "west"] * 12]
    vocab_literals = candidate_literals(vocab_cases)
    vocab_rules = decision_list(vocab_cases, vocab_literals, min_confidence=0.3)
    vocab = verify_decision_list(vocab_rules, train=vocab_cases, held_out=vocab_cases, literals=vocab_literals)

    return {
        "real_task": {"rules": [repr(r) for r in rules.rules], "default": rules.default,
                      "held_out": real.held_out, "train": real.train, "floor": real.floor,
                      "random_control": round(real.random, 4), "shifted_control": round(real.shifted, 4),
                      "renamed_identical": real.renamed_identical, "adopted": real.adopted,
                      "reasons": list(real.verdict.reasons)},
        "vocabulary_reading_task": {"held_out": vocab.held_out, "random_control": round(vocab.random, 4),
                                    "renamed_identical": vocab.renamed_identical, "adopted": vocab.adopted,
                                    "reasons": list(vocab.verdict.reasons)},
    }


def main() -> None:
    report = {
        "date": time.strftime("%Y-%m-%d"),
        "python": sys.version.split()[0],
        "model_calls": 0,
        "act_benchmark": act_benchmark(),
        "compositional_in_lexicon": compositional(VILLAGE, label="in_lexicon"),
        "compositional_open_vocabulary": compositional(ENGLISH, OPEN_VOCABULARY, label="open_vocabulary"),
        "discourse": discourse_demo(),
        "induction": induction(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1, default=str))
    acts = report["act_benchmark"]
    comp = report["compositional_in_lexicon"]
    open_vocab = report["compositional_open_vocabulary"]
    ind = report["induction"]
    print(f"act benchmark   grammar {acts['grammar_correct']}/{acts['cases']} ({acts['grammar_accuracy']:.1%})"
          f"   regexes {acts['regex_correct']}/{acts['cases']} ({acts['regex_accuracy']:.1%})"
          f"   p50 {acts['ms_per_utterance']['p50']}ms")
    print(f"compositional    in-lexicon {comp['grammar_correct']}/{comp['cases']} ({comp['grammar_accuracy']:.1%})"
          f"   covered {comp['fully_covered']}/{comp['cases']}   round-trips {comp['round_trips']}/{comp['cases']}"
          f"   regexes unknown/wrong on {comp['regex_unknown_or_wrong']}/{comp['cases']}")
    print(f"                 OPEN VOCAB {open_vocab['grammar_correct']}/{open_vocab['cases']} "
          f"({open_vocab['grammar_accuracy']:.1%})   covered {open_vocab['fully_covered']}/{open_vocab['cases']}"
          f"   round-trips {open_vocab['round_trips']}/{open_vocab['cases']}")
    print(f"induction       real task adopted={ind['real_task']['adopted']} "
          f"held-out {ind['real_task']['held_out']:.3f} vs random {ind['real_task']['random_control']:.3f} "
          f"floor {ind['real_task']['floor']:.3f}")
    print(f"                vocabulary task adopted={ind['vocabulary_reading_task']['adopted']} "
          f"({'; '.join(ind['vocabulary_reading_task']['reasons']) or 'no reasons'})")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
