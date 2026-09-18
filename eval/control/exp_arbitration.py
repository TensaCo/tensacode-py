"""FACULTY 2 — priority arbitration.

Prediction (stated before the measurement): against today's static ordering, arbitration lowers
the waiting time of a cheap urgent goal without thrashing (switches per completed goal stays near
1), and aging keeps the expensive goal from starving.

Two measurements:

1. the policy itself, on a synthetic goal set, against the ordering the assistant actually uses
   today (arrival order, run to completion, never switch). Environment ours, grader ours.
2. the policy inside the assistant, deciding which of two set-aside questions comes back first.
   This is where the honest limitation shows: ``hear`` can weigh a request's *cost* (steps left
   in its procedure) but nothing in the parser produces value or urgency, so in the live agent
   arbitration is cost-first and nothing else.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field

import tensacode as tc
from tensacode import control as C

from eval.control.exp_resumption import AMBIGUOUS
from eval.control.harness import Conversation


@dataclass
class Job:
    name: str
    arrives: int
    steps: int
    value: float = 1.0
    urgency: float = 0.0
    deadline: float | None = None
    left: int = field(init=False)

    def __post_init__(self) -> None:
        self.left = self.steps

    @property
    def ref(self) -> tc.Ref:
        return tc.Ref(f"goal:{self.name}")


def simulate(jobs: list[Job], *, stance: C.Stance | None, ticks: int = 400) -> dict:
    """One unit of work per tick, to whichever goal the policy picks. stance=None is today's rule."""
    jobs = [Job(j.name, j.arrives, j.steps, j.value, j.urgency, j.deadline) for j in jobs]
    mind = tc.Store()
    by_ref = {j.ref: j for j in jobs}
    live: list[Job] = []
    current: tc.Ref | None = None
    switches, first_touch, finished, worked = 0, {}, {}, {}
    for tick in range(ticks):
        for j in jobs:
            if j.arrives == tick:
                C.declare(mind, j.ref, what=j.name, source="eval:arbitration", value=j.value,
                          cost_to_go=float(j.left), deadline=j.deadline)
                C.weigh(mind, j.ref, source="eval:arbitration", urgency=j.urgency)
                C.waiting_since(mind, j.ref, float(tick), "waiting its turn", source="eval:arbitration")
                live.append(j)
        live = [j for j in live if j.left > 0]
        if not live:
            if all(j.left == 0 for j in jobs):
                break
            continue
        if stance is None:  # today: the oldest live goal, worked to completion, never switched
            pick = by_ref[current] if current is not None and by_ref[current].left > 0 else min(live, key=lambda j: (j.arrives, j.name))
            chosen = pick.ref
        else:
            chosen = C.arbitrate(mind, stance=stance, now=float(tick), current=current,
                                 among=[j.ref for j in live], source="eval:arbitration").goal
        if chosen != current:
            switches += 1 if current is not None else 0
            current = chosen
        job = by_ref[chosen]
        first_touch.setdefault(job.name, tick - job.arrives)
        job.left -= 1
        worked[job.name] = worked.get(job.name, 0) + 1
        C.weigh(mind, job.ref, source="eval:arbitration", cost_to_go=float(job.left))
        if job.left == 0:
            finished[job.name] = tick + 1 - job.arrives
            C.settle(mind, job.ref, C.DONE, source="eval:arbitration")
            current = None
    done = len(finished)
    return {
        "completed": done,
        "switches": switches,
        "switches_per_completed_goal": round(switches / done, 2) if done else None,
        "latency": finished,  # ticks from arrival to completion
        "wait_before_first_touch": first_touch,
        "mean_latency": round(sum(finished.values()) / done, 2) if done else None,
        "worst_latency": max(finished.values()) if finished else None,
        "starved": [j.name for j in jobs if j.name not in finished],
    }


STANCES = {
    "today (arrival order, no switching)": None,
    "arbitrate (library default)": C.Stance(),
    "arbitrate (assistant's stance)": C.Stance(stickiness=0.35, aging=0.25, cost_weight=0.05),
    "arbitrate, no stickiness": C.Stance(stickiness=0.0),
    "arbitrate, no aging": C.Stance(aging=0.0),
}

#: designed against: the directive's own case, one cheap-and-urgent against one expensive-and-important
CHEAP_VS_EXPENSIVE = [
    Job("expensive_important", arrives=0, steps=20, value=2.0),
    Job("cheap_urgent", arrives=3, steps=2, value=1.0, urgency=3.0),
]

#: held out: written after the policy was fixed — six goals, staggered arrivals, one long and
#: dull (the starvation trap) and one with a deadline
MIXED = [
    Job("long_and_dull", arrives=0, steps=30, value=0.8),
    Job("medium", arrives=2, steps=8, value=1.0),
    Job("quick_a", arrives=4, steps=2, value=1.0, urgency=1.0),
    Job("quick_b", arrives=5, steps=2, value=1.0, urgency=1.0),
    Job("deadline_soon", arrives=6, steps=5, value=1.0, deadline=14.0),
    Job("quick_c", arrives=20, steps=2, value=1.0, urgency=2.0),
]


#: held out, and built to expose a failure rather than to flatter: cheap goals keep arriving, so
#: a cost-first rule can pass over the long one forever. This is the only test where aging can
#: show its worth; in a closed world every goal eventually completes whatever the policy.
STREAM = [Job("long_and_dull", arrives=0, steps=25, value=1.0)] + [
    Job(f"cheap_{i:02d}", arrives=2 + 3 * i, steps=2, value=1.0) for i in range(30)
]


def conversation_arm() -> dict:
    """Two questions set aside; which one does the assistant pick back up, and does either starve?"""
    c = Conversation(AMBIGUOUS)  # report.pdf exists twice, so delete has to ask too
    script = [
        "tidy up my desktop",                         # question 1 (20 steps left in its procedure)
        "delete report.pdf",                          # suspends question 1, asks question 2 (19 left)
        "how many words are in ~/Desktop/notes.txt",  # suspends question 2 for a cheap fresh job
        "carry on",                                   # which question comes back first?
        "1",                                          # answer it
        "carry on",                                   # does the other one come back at all?
        "1",
    ]
    turns = [c.say(t) for t in script]
    resumed = [r for t in turns for r in t.replies if r.startswith("Back to it")]
    weights = {}
    for req in c.requests():
        ref = tc.Ref(req)
        weights[req] = {"words": agentwords(c, ref), "cost_to_go": C._one(c.mind, ref, "goal:cost_to_go"),
                        "value": C._one(c.mind, ref, "goal:value"), "urgency": C._one(c.mind, ref, "goal:urgency"),
                        "chosen_because": C._one(c.mind, ref, "goal:chosen_because"),
                        "status": c.status(req), "resumed": None}
    return {"script": script, "resumption_order": [r[:60] for r in resumed], "weights": weights,
            "turns": [{"said": t.text, "intentions": t.intentions, "replies": t.replies} for t in turns],
            "limitation": "value and urgency are never set by hear(): nothing in the parser produces them, "
                          "so live arbitration is cost-first only"}


def agentwords(c: Conversation, ref: tc.Ref) -> str:
    from examples.browser_agents.assistant import agent
    return str(agent.one(c.mind, ref, "words"))


def main() -> None:
    out = {"faculty": "priority arbitration",
           "prediction": "lower waiting for a cheap urgent goal, switches per completed goal near 1, no starvation",
           "provenance": {"environment": "ours (synthetic goal set; assistant conversation for the second arm)",
                          "grader": "ours", "cheap_vs_expensive": "designed against", "mixed": "held out", "stream": "held out"}}
    for name, jobs in (("cheap_vs_expensive", CHEAP_VS_EXPENSIVE), ("mixed", MIXED), ("stream", STREAM)):
        out[name] = {}
        print(f"\n=== {name}")
        for label, stance in STANCES.items():
            r = simulate(jobs, stance=stance)
            out[name][label] = r
            print(f"  {label:38} switches/goal={r['switches_per_completed_goal']} "
                  f"mean_latency={r['mean_latency']} worst={r['worst_latency']} starved={r['starved']}")
            worst = sorted(r["latency"].items(), key=lambda kv: -kv[1])[:3]
            print(f"     {'':36} worst three: {worst}")
    out["in_the_assistant"] = conversation_arm()
    print("\n=== in the assistant")
    for line in out["in_the_assistant"]["resumption_order"]:
        print("   back to:", line.replace("\n", " / "))
    for req, w in out["in_the_assistant"]["weights"].items():
        print(f"   {req:14} cost_to_go={w['cost_to_go']} value={w['value']} urgency={w['urgency']} "
              f"status={w['status']} because={w['chosen_because']!r}")
    path = "eval/results/control_arbitration.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=1, default=str)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    sys.exit(main())
