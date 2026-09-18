"""FACULTY 1 — interruption and resumption.

Prediction (stated before the measurement): today the assistant cannot resume an interrupted
request; with a task set it completes both requests without redoing finished steps. Falsified if
resumption requires re-running completed steps — that would be restart, not resumption.

What is counted is *redone work*: a step of a goal that executes again after already having
executed for that goal (see :func:`eval.control.harness.redone`, which reads step identity —
procedure, frame depth, program counter — back off the store). A goal is keyed by the words the
user said, so restating a dropped request counts as the same goal: that is exactly the cost of
"drop it and apologise".
"""

from __future__ import annotations

import json
import sys

from eval.control.harness import HOME, TREE, Conversation, redone
from eval.control.legacy import as_before

#: two files with the same name, in two places the assistant looks: resolve has to ask
AMBIGUOUS = dict(TREE, **{f"{HOME}/Desktop/report.pdf": "desktop copy\n"})


def episode(script: list[str], tree: dict) -> dict:
    c = Conversation(tree)
    turns = [c.say(text) for text in script]
    statuses = {r: c.status(r) for r in c.requests()}
    again = redone(c)
    return {
        "turns": [{"said": t.text, "intentions": t.intentions, "commands": t.commands, "replies": t.replies} for t in turns],
        "commands_total": sum(len(t.commands) for t in turns),
        "steps_redone": sum(again.values()),
        "steps_redone_by_goal": again,
        "statuses": statuses,
        "requests_done": sum(1 for v in statuses.values() if v == "done"),
        "requests_dropped": sum(1 for v in statuses.values() if v == "dropped"),
        "apologised": [r for t in turns for r in t.replies if "skipping" in r],
        "resumption_replies": [r for t in turns for r in t.replies if r.startswith("Back to it")],
        "not_understood": [r for t in turns for r in t.replies if "didn't understand" in r],
        "goal_finished": any("Grouped" in r or "Deleted" in r for t in turns for r in t.replies),
    }


CASES = {
    # designed against: the case the mechanism was built for
    "organize_interrupted_by_create": dict(
        tree=TREE,
        now=["tidy up my desktop", "make a folder called plans on the desktop", "carry on", "1"],
        # the old layer had no way to refer to a set-aside goal, so its honest recovery is to say
        # the whole thing again; that restatement is what the redone-work number prices
        before=["tidy up my desktop", "make a folder called plans on the desktop", "carry on", "tidy up my desktop", "1"],
        carry_on_only=["tidy up my desktop", "make a folder called plans on the desktop", "carry on"],
    ),
    # held out: written after the mechanism was finished — different procedure (the question
    # comes from resolve, not clarify_goal), different interrupter, different resumption words
    "ambiguous_delete_interrupted_by_count": dict(
        tree=AMBIGUOUS,
        now=["delete report.pdf", "how many words are in ~/Desktop/notes.txt", "where were we", "2"],
        before=["delete report.pdf", "how many words are in ~/Desktop/notes.txt", "where were we", "delete report.pdf", "2"],
        carry_on_only=["delete report.pdf", "how many words are in ~/Desktop/notes.txt", "where were we"],
    ),
}

PROVENANCE = {"organize_interrupted_by_create": "designed against",
              "ambiguous_delete_interrupted_by_count": "held out"}


def main() -> None:
    out = {"faculty": "interruption and resumption",
           "prediction": "today it cannot resume; with a task set both requests complete with no redone steps",
           "provenance": {"environment": "ours (fake shell, real control layer and procedures)",
                          "grader": "ours (step executions read off the store)", "cases": PROVENANCE}}
    for name, case in CASES.items():
        with as_before():
            before = episode(case["before"], case["tree"])
            carry = episode(case["carry_on_only"], case["tree"])
        after = episode(case["now"], case["tree"])
        out[name] = {"before_restated": before, "before_carry_on_only": carry, "after": after}
        print(f"\n=== {name} ({PROVENANCE[name]})")
        for label, run in (("before, restated", before), ("before, “carry on” only", carry), ("after", after)):
            print(f"  {label:26} done={run['requests_done']} dropped={run['requests_dropped']} "
                  f"commands={run['commands_total']} steps_redone={run['steps_redone']} "
                  f"goal_finished={run['goal_finished']}")
            for t in run["turns"]:
                print(f"      · {t['said']!r:50} {t['intentions']}")
    path = "eval/results/control_resumption.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    sys.exit(main())
