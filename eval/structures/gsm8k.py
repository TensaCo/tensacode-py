"""Measurement 1: do typed quantities make arithmetic word problems tractable?

The rule arm scored 0.0% on GSM8K in `docs/revival/13-schema-brittleness.md`, and the
brittleness taxonomy blamed missing representation. This asks a narrower question: with a
quantity type, unit checking and recorded derivations, how far does a symbolic reader get,
and *where* does it stop — reading the sentence, composing the arithmetic, or needing a
fact the problem never states.

The solver is deliberately cue-driven rather than a search over all arithmetic. A search
that tries every composition and keeps whichever lands on a plausible number would score
better and mean nothing: it would be selecting for the answer, not deriving it. So the only
compositions attempted are ones a cue in the text licenses, the unit of the result must
match the unit the question asks for, and anything else is a refusal.

    python -m eval.structures.gsm8k --n 300
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from tensorcode.outcomes import Unknown
from tensorcode.quantity import Quantity, Unit, add, div, mul, normalize_unit, sub
from tensorcode.records import Ref, Store
from tensorcode.semantics_bridge import Mention, quantities_in_text

OUT = Path(__file__).resolve().parents[2] / "eval" / "results" / "structures_gsm8k.json"

TOTAL_CUES = ("in total", "altogether", "in all", "combined", "total number", "how many .* and")
LEFT_CUES = ("left", "remain", "remaining", "still have", "how many more", "difference", "fewer", "less than")
EACH_CUES = ("each", "per ", "every", "apiece", "a piece")
QUESTION = re.compile(r"how (?:many|much)\s+([a-z-]+(?:\s+[a-z-]+)?)|what is the (?:total|cost|price)", re.I)


@dataclass
class Attempt:
    """One problem, what was read from it, and where the attempt stopped."""

    index: int
    answer: float | None
    gold: float
    correct: bool
    stage: str  # where it stopped: read | question | compose | computed
    reason: str = ""
    external_constants: tuple[float, ...] = ()  # numbers the gold working uses that the text never states
    quantities: int = 0
    steps: tuple[str, ...] = ()


def gold_answer(row: dict) -> tuple[float, tuple[float, ...]]:
    """The gold number, and the numbers its working uses that the question does not state.

    GSM8K's solutions carry their arithmetic in ``<<...>>`` annotations. A constant that
    appears there but nowhere in the question is world knowledge the reader was expected to
    supply ("a week has 7 days"), which is the honest explanation for a whole class of
    failures and is worth counting rather than lumping in with parse errors.
    """
    text = row["answer"]
    final = float(text.rsplit("####", 1)[1].strip().replace(",", ""))
    known = {_num(m) for m in re.findall(r"\d+(?:\.\d+)?", row["question"].replace(",", ""))}
    external: list[float] = []
    # walk the working in order: an operand is external knowledge only if the question never
    # stated it AND no earlier step produced it. Counting every number in the working would
    # call each intermediate result "world knowledge", which is what a first pass did.
    for expr in re.findall(r"<<([^>]*)>>", text):
        left, _, right = expr.partition("=")
        operands = [_num(m) for m in re.findall(r"\d+(?:\.\d+)?", left.replace(",", ""))]
        for value in operands:
            if value not in known and value not in (0.0, 1.0, 2.0, 100.0):
                external.append(value)
        known.update(operands)
        known.update(_num(m) for m in re.findall(r"\d+(?:\.\d+)?", right.replace(",", "")))
    return final, tuple(sorted(set(external)))


def _num(s: str) -> float:
    return float(s)


def question_unit(text: str) -> tuple[str | None, str]:
    """What the question asks for, as a unit — the check that stops a wrong-dimension answer."""
    tail = text.rsplit("?", 1)[0]
    tail = tail[tail.rfind(".") + 1:] if "." in tail else tail
    match = QUESTION.search(tail)
    if not match:
        return None, "no_question_phrase"
    words = (match.group(1) or "").split()
    for word in words:
        unit = normalize_unit(word)
        if unit and word.isalpha() and word not in ("does", "did", "will", "would", "the", "of", "is", "are", "do"):
            return unit, ""
    return None, "no_unit_in_question"


#: dimensions of things that are counted or held, where a negative result is not a number
#: you can have. A negative count is a sign the composition was wrong, not an answer.
def _impossible(got: Quantity) -> bool:
    return got.value < 0 and all(d.startswith("count:") or d in ("item", "currency", "volume", "mass", "length", "time")
                                 for d, _ in got.dimension)


def _search(mentions: list[Mention], want: str, lowered: str, steps: list[str]) -> Quantity | Unknown:
    """A bounded search over two- and three-term compositions, chosen by unit and by cue.

    The gold answer is never consulted: candidates are ordered by a fixed preference (use
    the quantities the question mentions, prefer the operation the wording licenses) and
    the first one whose unit matches what was asked is returned. Ties that the wording does
    not settle are refused, because picking one would be guessing.
    """
    same = [m for m in mentions if str(m.quantity.unit) == want]
    ops: list[tuple[str, Any]] = []
    if any(re.search(cue, lowered) for cue in LEFT_CUES):
        ops.append(("−", sub))
    if any(re.search(cue, lowered) for cue in TOTAL_CUES):
        ops.append(("+", add))
    if not ops:  # no cue: only a composition licensed by the units themselves
        ops = []
    candidates: list[tuple[str, Quantity]] = []
    for symbol, op in ops:
        for i, a in enumerate(same):
            for j, b in enumerate(same):
                if i == j:
                    continue
                got = op(a.quantity, b.quantity)
                if isinstance(got, Unknown) or str(got.unit) != want or _impossible(got):
                    continue
                candidates.append((f"{a.quantity} {symbol} {b.quantity} = {got}", got))
        if symbol == "+" and len(same) > 2:  # "in total" over everything stated
            running: Quantity | Unknown = same[0].quantity
            for mention in same[1:]:
                running = add(running, mention.quantity)
                if isinstance(running, Unknown):
                    break
            if not isinstance(running, Unknown) and str(running.unit) == want:
                candidates.append((" + ".join(str(m.quantity) for m in same) + f" = {running}", running))
    if not candidates:
        return Unknown("no_licensed_composition", f"{len(mentions)} quantities, none composable into {want}")
    distinct = {round(q.value, 9) for _, q in candidates}
    if len(distinct) > 1:
        return Unknown("ambiguous_composition", f"{len(distinct)} compositions give different answers in {want}")
    steps.append(candidates[0][0])
    return candidates[0][1]


def solve(text: str, mind: Store, source: Ref) -> tuple[Quantity | Unknown, list[str]]:
    """Attempt the arithmetic a cue licenses, and refuse otherwise."""
    mentions = quantities_in_text(text)
    if not mentions:
        return Unknown("no_quantities", "nothing numeric read from the problem"), []
    want, why = question_unit(text)
    if want is None:
        return Unknown("no_question_unit", why), []
    steps: list[str] = []
    lowered = text.lower()
    matching = [m for m in mentions if str(m.quantity.unit) == want]
    rates = [m for m in mentions if m.per]
    counts = [m for m in mentions if not m.per]

    # a rate times a count, when the text says "each" or "per"
    if rates and any(re.search(cue, lowered) for cue in EACH_CUES):
        for rate in rates:
            for count in counts:
                got = mul(count.quantity, rate.quantity)
                if str(got.unit) == want:
                    steps.append(f"{count.quantity} × {rate.quantity} = {got}")
                    return got, steps

    if len(matching) >= 2:
        got = _search(mentions, want, lowered, steps)
        if not isinstance(got, Unknown):
            return got, steps
        if got.reason == "ambiguous_composition":
            return got, steps

    if len(matching) == 1 and len(mentions) >= 2:
        other = next((m for m in mentions if m is not matching[0]), None)
        if other is not None and other.quantity.unit.dimensionless:
            product = mul(matching[0].quantity, other.quantity)
            if not _impossible(product):
                steps.append(f"{matching[0].quantity} × {other.quantity} = {product}")
                return product, steps

    return Unknown("no_licensed_composition", f"{len(mentions)} quantities, none composable into {want}"), steps


def run(rows: Iterable[dict]) -> list[Attempt]:
    out: list[Attempt] = []
    for index, row in enumerate(rows):
        gold, external = gold_answer(row)
        mind, source = Store(), Ref("obs:problem")
        got, steps = solve(row["question"], mind, source)
        if isinstance(got, Unknown):
            stage = {"no_quantities": "read", "no_question_unit": "question"}.get(got.reason, "compose")
            out.append(Attempt(index, None, gold, False, stage, got.reason, external,
                               len(quantities_in_text(row["question"])), tuple(steps)))
            continue
        correct = abs(got.value - gold) < 1e-6
        out.append(Attempt(index, got.value, gold, correct, "computed", "" if correct else "wrong_value",
                           external, len(quantities_in_text(row["question"])), tuple(steps)))
    return out


def summarize(attempts: list[Attempt]) -> dict[str, Any]:
    n = len(attempts)
    answered = [a for a in attempts if a.answer is not None]
    correct = [a for a in answered if a.correct]
    needed_knowledge = [a for a in attempts if a.external_constants]
    stages: dict[str, int] = {}
    for a in attempts:
        if not a.correct:
            stages[a.stage] = stages.get(a.stage, 0) + 1
    solvable = [a for a in attempts if not a.external_constants]
    return {
        "n": n,
        "coverage": round(len(answered) / n, 4) if n else 0.0,
        "accuracy_overall": round(len(correct) / n, 4) if n else 0.0,
        "accuracy_when_answered": round(len(correct) / len(answered), 4) if answered else 0.0,
        "wrong_when_answered": len(answered) - len(correct),
        "failures_by_stage": stages,
        "items_needing_unstated_constants": len(needed_knowledge),
        "share_needing_unstated_constants": round(len(needed_knowledge) / n, 4) if n else 0.0,
        "accuracy_on_items_not_needing_them": round(
            sum(1 for a in solvable if a.correct) / len(solvable), 4) if solvable else 0.0,
        "examples_correct": [{"i": a.index, "steps": list(a.steps), "answer": a.answer} for a in correct[:5]],
        "examples_wrong": [{"i": a.index, "steps": list(a.steps), "answer": a.answer, "gold": a.gold} for a in answered if not a.correct][:5],
    }


def main() -> None:
    from datasets import load_dataset

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    data = load_dataset("openai/gsm8k", "main", split=args.split)
    rows = [data[i] for i in range(min(args.n, len(data)))]
    started = time.perf_counter()
    attempts = run(rows)
    report = summarize(attempts)
    report["seconds"] = round(time.perf_counter() - started, 2)
    report["ms_per_item"] = round(1000 * report["seconds"] / max(1, len(rows)), 3)
    report["arm"] = "tensorcode quantities (no model)"
    report["baselines"] = {"rule_arm_before": 0.0, "local_model_qwen3_8b": 0.813}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if not k.startswith("examples")}, indent=1))


if __name__ == "__main__":
    main()
