"""Error repair and agency attribution, measured in computerworld.

Two faculties, one harness, because both need the same thing: a real environment that
fails in real ways and can change without being asked.

**Repair.** The environment supplies its own failure modes — no injection needed. Its
documented gaps (``eval/results/computerworld_gaps.json``) include a `find -name` that
ignores its pattern, a missing `stat -c`, a missing `du`/`df`/`which`, and unsupported
`2>/dev/null`. The baseline is today's behaviour in the assistant's procedures: run the
command, read the error, report it, stop — no second attempt of any kind. The treatment is
:class:`tensacode.metacognition.Monitor`: name the failure, choose a repair, and never
re-issue an action that already failed the same way.

**Agency.** `CwWorld.shell` is a privileged channel the *agent* does not have, so the
harness can change the world between the agent's own actions. That supplies the
spontaneous dynamics ``docs/revival/17`` said were missing (it concluded causal
discrimination was untestable without them). Each observed change is then attributed by
:func:`tensacode.metacognition.attribute` using the action's own predicted effect as
efference copy, against two controls: the agent acts while the world is frozen, and the
world moves while the agent does nothing.

Provenance: environment is the user's computerworld engine (not ours); outcomes are read
from the engine's own filesystem and terminal, not from any model's report; the failure
modes are the engine's real gaps rather than faults we introduced. Deterministic by seed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.browser_agents.worlds import desktop_world  # noqa: E402
from examples.browser_agents.worlds.runtime import CwWorld  # noqa: E402
from tensacode.metacognition import Monitor, attribute, surprising  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "eval" / "results" / "metacognition_repair_agency.json"

#: (goal, first attempt, a genuinely different second attempt, how to tell it worked)
#: every first attempt here fails because of a real engine gap, not because we broke it
TASKS = [
    ("find a file by name", "find /home/agent -name handbook.txt", "ls -1 /home/agent/Documents", "handbook.txt"),
    ("check whether a path is a directory", "stat -c '%F' /home/agent/Documents", "ls -1p /home/agent | grep Documents/", "Documents/"),
    ("measure a folder", "du -sh /home/agent/Documents", "ls -1 /home/agent/Documents", "handbook.txt"),
    ("locate a program", "which git", "ls -1 /usr/bin", "git"),
    ("read a file quietly", "cat /home/agent/nope.txt 2>/dev/null", "ls -1 /home/agent", "Documents"),
]


def classify(sh, expect: str) -> str | None:
    """Name the failure from what the environment actually said."""
    if sh.exit_code != 0 or sh.stderr.strip():
        return "error_output"
    if not sh.stdout.strip():
        return "no_effect"
    if expect not in sh.stdout:
        return "wrong_result"  # it ran, and the answer is not the one asked for
    return None


def repair_trial(world: CwWorld) -> dict:
    """Baseline (report and stop) against the repair repertoire, on the engine's own gaps."""
    baseline = {"attempts": 0, "solved": 0, "repeats": 0}
    treated = {"attempts": 0, "solved": 0, "repeats": 0, "repairs": []}

    for goal, first, different, expect in TASKS:
        sh = world.shell(first)
        failure = classify(sh, expect)
        baseline["attempts"] += 1
        baseline["solved"] += 0 if failure else 1

        m = Monitor()
        treated["attempts"] += 1
        sh = world.shell(first)
        failure = classify(sh, expect)
        if not failure:
            treated["solved"] += 1
            continue
        m.note(first, failure, sh.stderr.strip() or sh.stdout.strip()[:60])
        plan = m.repair(first, failure)
        treated["repairs"].append({"goal": goal, "failed": first, "failure": failure, "repair": plan.kind, "why": plan.why})
        if plan.kind in ("retry", "retry_differently", "reperceive"):
            # the repertoire says "not the same act"; the different act is the caller's to supply
            action = first if plan.kind == "retry" else different
            if action == first:
                treated["repeats"] += 1
            treated["attempts"] += 1
            sh2 = world.shell(action)
            if classify(sh2, expect) is None:
                treated["solved"] += 1
            else:
                m.note(action, classify(sh2, expect) or "wrong_result")
                treated["repairs"][-1]["second"] = m.repair(action, classify(sh2, expect) or "wrong_result").kind
    return {"baseline_report_and_stop": baseline, "with_repair_repertoire": treated,
            "note": "the first attempt of every task fails because of a documented engine gap, not an injected fault"}


def agency_trial(world: CwWorld) -> dict:
    """Attribute change to self or world, with both controls."""

    def aspects() -> dict:
        """What the agent can see of the world's state, read through the engine."""
        home = world.entries("/home/agent") or []
        desktop = world.entries("/home/agent/Desktop") or []
        return {"home_entries": len(home), "desktop_entries": len(desktop),
                "desktop_names": ",".join(sorted(desktop))}

    rows = []

    # 1. the agent acts, and only its own action changes anything
    before = aspects()
    world.shell("mkdir -p /home/agent/Desktop/mine")
    got = attribute(aspects(), before=before,
                    predicted={"desktop_entries": before["desktop_entries"] + 1,
                               "desktop_names": ",".join(sorted(set((world.entries("/home/agent/Desktop") or [])))) },
                    acted=True)
    rows.append({"case": "agent acts, world frozen", "expected": "self", "attributions": got,
                 "correct": all(v == "self" for v in got.values()) and bool(got)})

    # 2. the agent acts AND the world moves on its own in the same interval
    before = aspects()
    world.shell("mkdir -p /home/agent/Desktop/also-mine")          # the agent's own act
    world.shell("printf 'log line\\n' > /home/agent/world-wrote-this.txt")  # nobody asked for this
    after = aspects()
    predicted = {"desktop_entries": before["desktop_entries"] + 1,
                 "desktop_names": ",".join(sorted(set((world.entries("/home/agent/Desktop") or []))))}
    got = attribute(after, before=before, predicted=predicted, acted=True)
    rows.append({"case": "agent acts, world also moves", "expected": "self for the desktop, world for home",
                 "attributions": got,
                 "correct": got.get("desktop_entries") == "self" and got.get("home_entries") == "world",
                 "surprising": list(surprising(got))})

    # 3. control: the world moves while the agent does nothing
    before = aspects()
    world.shell("printf 'again\\n' > /home/agent/world-wrote-this-too.txt")
    got = attribute(aspects(), before=before, predicted={}, acted=False)
    rows.append({"case": "world moves, agent idle", "expected": "world", "attributions": got,
                 "correct": bool(got) and all(v == "world" for v in got.values())})

    # 4. control: the agent acts and nothing at all changes
    before = aspects()
    world.shell("true")
    got = attribute(aspects(), before=before, predicted={"desktop_entries": before["desktop_entries"]}, acted=True)
    rows.append({"case": "agent acts, nothing changes", "expected": "no attribution at all",
                 "attributions": got, "correct": got == {}})

    return {"cases": rows, "correct": sum(r["correct"] for r in rows), "n": len(rows),
            "note": "world changes are made through the owner's privileged shell, a channel the agent does not have"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    world = CwWorld(desktop_world(), args.seed)
    report = {
        "provenance": {"environment_author": "the user's computerworld engine",
                       "grader": "the engine's own filesystem and terminal output",
                       "failures": "the engine's documented gaps (eval/results/computerworld_gaps.json), not injected",
                       "determinism": world.state_hash()},
        "repair": repair_trial(world),
        "agency": agency_trial(CwWorld(desktop_world(), args.seed)),
    }
    base = report["repair"]["baseline_report_and_stop"]
    treat = report["repair"]["with_repair_repertoire"]
    report["prediction"] = {
        "repair_solves_more_than_report_and_stop": treat["solved"] > base["solved"],
        "baseline_solved": f"{base['solved']}/{base['attempts']}",
        "repaired_solved": f"{treat['solved']}/{len(TASKS)}",
        "identical_repeats": treat["repeats"],
        "agency_correct": f"{report['agency']['correct']}/{report['agency']['n']}",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1))
    print(f"wrote {OUT}")
    print(json.dumps(report["prediction"], indent=1))
    for r in report["agency"]["cases"]:
        print(f"  agency [{'ok' if r['correct'] else 'WRONG'}] {r['case']}: {r['attributions']}")
    for r in treat["repairs"]:
        print(f"  repair {r['goal']}: {r['failure']} -> {r['repair']}")


if __name__ == "__main__":
    main()
