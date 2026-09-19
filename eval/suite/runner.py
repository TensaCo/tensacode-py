"""Run a task against a subject, judge every item, record the result.

    python -m eval.suite.runner --tasks language.* --subject agent:learned:desktop+vision
    python -m eval.suite.runner --all --split dev --controls

Nothing here knows any particular benchmark: it walks the registry, and a task that has
no data says so instead of failing the run.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path[:0] = [str(Path(__file__).parents[2] / "src"), str(Path(__file__).parents[2])]

from eval.suite import store  # noqa: E402
from eval.suite.core import Judgement, Task, metrics, registry  # noqa: E402
from eval.suite.subjects import AgentSubject, named  # noqa: E402


def run_task(task: Task, subject, split: str, *, limit: int = 0, show: bool = False) -> dict | None:
    if not task.dataset.available():
        print(f"  {task.id}: data missing ({task.dataset.fetch_hint})")
        return None
    items = list(task.items(split))
    if limit:
        items = items[:limit]
    if not items:
        print(f"  {task.id}: no items in split {split}")
        return None
    judgements: list[Judgement] = []
    detail = []
    crashed = 0
    t0 = time.perf_counter()
    for item in items:
        runner = subject.fresh() if isinstance(subject, AgentSubject) else subject
        try:
            response = task.execute(runner, item)
            crash = ""
        except Exception as exc:  # noqa: BLE001 - a crash is a result, but not an answer
            crash = f"{type(exc).__name__}: {exc}"
            response = type("R", (), {"text": f"__crash__ {crash}", "abstained": False, "detail": {}})()
        judgement = task.judge(item, response)
        if crash:
            # the judge saw a string like any other; the run knows better
            judgement = replace(judgement, answered=False, correct=None, score=None, errored=True, note=crash)
            crashed += 1
        judgements.append(judgement)
        detail.append({"id": item.id, "answered": judgement.answered, "correct": judgement.correct,
                       "score": judgement.score, "errored": judgement.errored, "reply": response.text[:300]})
        if show:
            print(f"    [{item.id}] {item.prompt.text[:60]!r} -> {response.text[:80]!r} "
                  f"answered={judgement.answered} correct={judgement.correct}")
    result = metrics(judgements)
    result["seconds"] = round(time.perf_counter() - t0, 1)
    row = store.record(task.id, subject.id, split, result, dataset=task.dataset.name,
                       license=task.dataset.license, items=detail, self_authored=task.self_authored,
                       notes=task.notes)
    print(f"  {task.id} [{subject.id}] {json.dumps({k: v for k, v in result.items() if v is not None})}")
    if crashed:
        first = next(j.note for j in judgements if j.errored)
        print(f"    !! {crashed}/{len(items)} items raised — this subject is broken here, not wrong: {first[:120]}")
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="*", help="glob over task ids, e.g. 'vision.*'")
    ap.add_argument("--subject", default="agent:learned:desktop+vision")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--controls", action="store_true", help="also run the task's declared controls")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()
    chosen = [t for tid, t in sorted(registry().items()) if args.all or fnmatch.fnmatch(tid, args.tasks)]
    if not chosen:
        sys.exit(f"no task matches {args.tasks!r}")
    subject = named(args.subject)
    if (why := getattr(subject, "unavailable", lambda: "")()):
        sys.exit(f"subject {subject.id} cannot be built here: {why}\n"
                 f"run a configuration that does not need it (e.g. --subject agent:{getattr(subject, 'reader', 'learned')}:vision) "
                 f"rather than recording a run of crashes")
    print(f"{len(chosen)} task(s), subject {subject.id}, split {args.split}")
    for task in chosen:
        if args.split not in task.splits:
            continue
        run_task(task, subject, args.split, limit=args.limit, show=args.show)
        if args.controls:
            for control in task.controls:
                run_task(task, named(control), args.split, limit=args.limit)


if __name__ == "__main__":
    main()
