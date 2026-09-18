"""Write docs/revival/18-cognitive-profile.md from eval/results/cognitive_profile.json.

    python -m eval.profile.report

The doc is generated so it cannot drift from the numbers. Prose that is a judgement
rather than a number lives in the axis ``reading`` fields in profile.py.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PROFILE = REPO / "eval" / "results" / "cognitive_profile.json"
DOC = REPO / "docs" / "revival" / "18-cognitive-profile.md"

STRENGTH = {"strong": "well grounded", "moderate": "partly grounded", "weak": "weakly grounded", "ungrounded": "**not grounded**"}


def main() -> None:
    p = json.loads(PROFILE.read_text())
    out: list[str] = []
    A = out.append

    A("# 18. How we measure general cognitive performance\n")
    A("Before this, the answer was: we did not. There were task scores with uneven provenance and no")
    A("common frame. This is the frame — a standing harness (`eval/profile/`) that recomputes from")
    A("scratch and writes `eval/results/cognitive_profile.json`, which generates this document.\n")
    A("**There is deliberately no single score.** A single number is exactly what invited the problems")
    A("the evidence audit found (§11): it can be raised by choosing an environment, a grader, or a")
    A("threshold. So every row below carries its floor, its control, who wrote the environment, who")
    A("graded it, whether the test was fixed before tuning, and its source file. Accuracies are")
    A("luck-corrected where the chance credit is known. An axis that cannot be grounded says so")
    A("rather than reporting a zero or quietly disappearing.\n")

    A(f"**{p['axes_grounded']} of {p['axes_total']} axes are grounded.** Ungrounded: "
      f"{', '.join('`' + k + '`' for k in p['axes_ungrounded']) or 'none'}.\n")

    # at-a-glance table
    A("## At a glance\n")
    A("| Axis | Question | Standing |")
    A("|---|---|---|")
    for a in p["axes"]:
        A(f"| `{a['key']}` | {a['question']} | {STRENGTH[a['verdict']]} |")
    A("")
    A("Read that column as *how much the evidence is worth*, not as how well we score. `horizon` is")
    A("well grounded and the news there is bad; `world_modeling` is partly grounded and the news is good.\n")

    A("**Where we are strong:** verifiable structured automation — long runs of short-horizon decisions")
    A("with zero model calls, checked against a simulator's own state; belief revision that re-perceives")
    A("rather than trusting memory; prediction of our own actions' effects, graded by the environment.\n")
    A("**Where we are weak:** open-domain knowledge and answer projection (§12, §13), genuinely long")
    A("tasks (nothing completes one), capability-aware abstention, and evidence integrity — most headline")
    A("rows are still graded by code we wrote in environments we wrote.\n")
    A("**Unmeasured:** sample efficiency as a rate (only two skills have ever been learned), and causal")
    A("discrimination (the intervention machinery runs but has not yet separated cause from co-occurrence).\n")

    for a in p["axes"]:
        A(f"## {a['key']}\n")
        A(f"*{a['question']}*\n")
        if not a["grounded"]:
            A(f"**Not grounded.** {a['ungrounded_reason']}\n")
        if a["reading"]:
            A(f"{a['reading']}\n")
        for m in a["measures"]:
            A(f"**{m['name']}**\n")
            A(f"- value: {m['value']}")
            if m["n"]:
                A(f"- n = {m['n']}" + (f", 95% CI {m['ci95']}" if m["ci95"] else ""))
            A(f"- floor: {m['floor']}")
            A(f"- control: {m['control']}")
            if m.get("chance_corrected"):
                A(f"- chance credit: {m['chance_corrected']}")
            A(f"- environment author: {m['env_author']} · grader: {m['grader']} · agent can influence grader: {m['agent_can_influence_grader']}")
            A(f"- held out: {m['heldout']}")
            A(f"- source: `{m['source']}`")
            if m["note"]:
                A(f"- note: {m['note']}")
            A("")

    A("## What we could not ground, and why\n")
    A("- **Sample efficiency as a rate.** Two skills have ever been learned (one adopted after succeeding")
    A("  on a request it was not learned from, one still on trial, one reuse in total). The mechanism and")
    A("  its adoption gate demonstrably run; a rate from n=2 would be noise. The teacher model was not")
    A("  running when this was computed, so no new learning could be measured.")
    A("- **Causal discrimination.** Interventions execute and the engine's fork/restore makes a real")
    A("  controlled experiment possible, but in the recorded run intervention found 14 links against")
    A("  correlation's 13, every aspect was judged `caused`, and the case built to show a co-occurring")
    A("  signal being *rejected* measured zero effect in both framings. The apparatus is there; the")
    A("  discrimination is not shown.")
    A("- **Robustness to unanticipated perturbation.** Every perturbation family we measure is one we")
    A("  built into our own apps. That is robustness to known unknowns.\n")

    A("## Two defects this harness found while being built\n")
    A("Both were found by driving the live assistant rather than by reading it, which is the pattern")
    A("that keeps holding in this project (§9, §11, §16).\n")
    A("1. **`copy` is broken.** `copy note.txt to <folder>` produces a run step with no command and")
    A("   raises `TypeError: 'NoneType' object is not subscriptable`. Reproduced in a fresh conversation")
    A("   in four separate turns, so it is the act itself and not a depth effect.")
    A("2. **A failed request ends the conversation.** After that crash, every later turn fails the same")
    A("   way — including turns that succeed in a fresh conversation — because the crashed request stays")
    A("   selected. There is no recovery short of a restart. This is the more serious of the two: one")
    A("   malformed act makes the agent permanently unusable rather than degrading one answer.\n")
    A("Both are in files this harness does not own, and are reported rather than patched.\n")

    A("## The three measurements that would most change our beliefs\n")
    A("1. **Capability-aware abstention, re-measured against the oracle gap.** §12 showed a router with")
    A("   perfect foresight gains only 1.3 and 2.0 points, so routing cannot be rescued by a better")
    A("   signal — but abstention itself is miscalibrated in both directions (54 wrongful refusals, 23")
    A("   answers to unanswerable questions). Measuring a calibrated head against those two error types")
    A("   separately would tell us whether `Unknown` is a real capability or a threshold. *Cost:* hours,")
    A("   reusing the existing 150/300 splits; no new environment.")
    A("2. **Horizon with a working long task.** Everything we know about long horizons comes from a")
    A("   0/8 result whose runs were throughput-bound at ~200 s per model call. Re-running the")
    A("   hidden-grader suite with a faster model would separate 'the architecture cannot do this' from")
    A("   'we never gave it enough steps'. *Cost:* a faster model or a day of local throughput work; the")
    A("   tasks and hidden checkers already exist.")
    A("3. **Prediction on an environment we did not write.** The prediction axis is the best-graded one")
    A("   we have, and it runs on a third-party engine — but on a task we authored. Pointing it at a")
    A("   public interactive benchmark would test whether expectation-learning is a property of the")
    A("   mechanism or of our world. *Cost:* the adapter, plus whatever the benchmark needs; the")
    A("   prediction machinery is unchanged.\n")

    A("## Running it\n")
    A("```sh")
    A("# live probes (each suite gets a fresh assistant instance: a crashed request is unrecoverable)")
    A("python -m examples.browser_agents.assistant.server --port 8773 --no-open &")
    A("python -m eval.profile.run_probes compositionality http://127.0.0.1:8773 probe_comp.json")
    A("python -m eval.profile.run_probes belief_revision   http://127.0.0.1:8774 probe_belief.json")
    A("python -m eval.profile.run_probes grounding         http://127.0.0.1:8775 probe_ground.json")
    A("# assemble, then regenerate this document")
    A("python -m eval.profile.profile --probes <dir-with-probe-json>")
    A("python -m eval.profile.report")
    A("```")

    DOC.write_text("\n".join(out) + "\n")
    print(f"wrote {DOC.relative_to(REPO)} ({len(out)} lines)")


if __name__ == "__main__":
    main()
