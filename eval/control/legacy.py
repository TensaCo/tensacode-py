"""The control layer as it was before this work, for a before/after measurement.

The assistant tree is untracked in git, so there is no commit to diff against. This is a
*reconstruction*, and it is faithful in exactly the ways that matter to the measurement:

* an interruption while a question is open **drops** the question and everything asked with it,
  and says so ("Okay, skipping …") — that was the only recovery the agent had;
* there is no "carry on": nothing in the control layer refers to a goal that is not current;
* the goal to pursue is ``fresh[0]`` — the order the requests arrived in, nothing else;
* ``open_requests`` has no suspended status, so a set-aside request is not even enumerable.

Everything else (hear, the decoders, the interpreter, the procedures) is the current code, so a
difference in the numbers below is a difference in the control layer and not in anything else.
"""

from __future__ import annotations

from contextlib import contextmanager

from examples.browser_agents.assistant import agent


def legacy_intentions(mind) -> list[object]:
    reqs = [r for r in agent.open_requests(mind) if agent.one(mind, r, "status") != "suspended"]
    awaiting = agent.one(mind, agent.ME, "awaiting")
    fresh = [r for r in reqs if agent.one(mind, r, "status") == "new"]
    running = [r for r in reqs if agent.one(mind, r, "status") == "running"]
    if running:
        return agent.body_intentions(mind, running[0])
    if awaiting is not None:
        asked_in = agent.one(mind, awaiting, "order")[0]
        answers = [r for r in fresh if agent.one(mind, r, "answers") == awaiting]
        if answers:
            return [agent.Advance(awaiting, agent.frame_of(mind, answers[0]),
                                  f"take “{agent.one(mind, answers[0], 'words')}” as the answer", priority=80)]
        if any(agent.one(mind, r, "order")[0] > asked_in for r in fresh):
            return [agent.Drop(awaiting, "you asked for something else, so I'll let that one go", priority=85)]
        return [agent.Finish("waiting for your answer", priority=100)]
    if fresh:
        current = fresh[0]
        return [agent.Start(current, f"start: {agent.one(mind, current, 'act')} “{agent.one(mind, current, 'words')[:60]}”", priority=70)]
    return [agent.Finish("all requests answered", priority=100)]


@contextmanager
def as_before():
    """Run a conversation with the old control layer."""
    now = agent.intentions
    agent.intentions = legacy_intentions
    try:
        yield
    finally:
        agent.intentions = now
