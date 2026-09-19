"""Acting: computer use, safety, and tool choice."""

from __future__ import annotations

from ..core import Dataset, Item, Judgement, Prompt, Response, Task, register
from ..data import heldout
from ..judges import attempted_only, changed_nothing

CONTROLS = ("control:abstain",)

for category, what in (("desktop_gui", "desktop tasks from OSWorld"),
                       ("shell_files", "file and shell requests (NL2Bash)"),
                       ("multi_step", "multi-step tasks (Mind2Web, OSWorld)")):
    register(Task(
        id=f"computer_use.{category}", area="computer use", what=what,
        dataset=heldout.dataset(category), judge=attempted_only, controls=CONTROLS,
        splits=("dev", "calibration", "test"),
        notes=("world-state graders are not written yet, so only engagement is recorded; "
               "most items need apps or shell features computerworld does not have")))

for category in ("general_knowledge", "arithmetic", "open_ended", "ambiguous", "image_questions",
                 "screen_questions", "conversation_facts"):
    register(Task(
        id=f"safety.no_change.{category}", area="safety", what=f"changes nothing when asked {category} questions",
        dataset=heldout.dataset(category), judge=changed_nothing, splits=("dev", "calibration", "test"),
        notes="correct == no capability with an effect was applied"))


def _missing(name: str, license: str, url: str, hint: str) -> Dataset:
    return Dataset(name=name, license=license, url=url, load=lambda split: [], available=lambda: False,
                   fetch_hint=hint)


register(Task(
    id="tools.function_calling", area="tools", what="picking the right tool and arguments, outside the desktop",
    dataset=_missing("Berkeley Function Calling Leaderboard", "Apache-2.0",
                     "https://github.com/ShishirPatil/gorilla",
                     "fetch BFCL's simple/multiple splits into ~/.cache/tensorcode/seeds"),
    judge=attempted_only, notes="the first tool-use measurement that is not computerworld"))

register(Task(
    id="tools.api_bank", area="tools", what="multi-call API use with a state-checking grader",
    dataset=_missing("API-Bank", "MIT", "https://github.com/AlibabaResearch/DAMO-ConvAI",
                     "fetch API-Bank level-1 dialogues"),
    judge=attempted_only))

def _native_items(split: str):
    """Desktop jobs written for this world, each with what must be true afterwards."""
    from examples.general_agent.tasks import JOBS

    return [Item(id=job.id, prompt=Prompt(text=job.prompt), gold=job.about, meta={"job": job})
            for job in JOBS]


def _native_available() -> bool:
    try:
        import computerworld  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def _run_native(subject, item: Item) -> Response:
    """Put the machine in the job's start state, ask, then look at the machine.

    The grade comes from the **owner** session afterwards, never from what the agent said it
    did: "I made the folder" is not evidence that a folder exists. A subject with no world —
    a control — cannot pass, which is the floor these jobs are measured against.
    """
    job = item.meta["job"]
    world = None
    if hasattr(subject, "build"):
        subject._agent = subject.build()          # so the start state is set up in *this* world
        world = subject._world
        for command in job.setup:
            world.shell(command)
    response = subject.respond(item.prompt)
    done = bool(world is not None and job.check(world, response.text))
    return Response(response.text, abstained=response.abstained,
                    detail={**dict(response.detail or {}), "done": done, "job": job.id})


def _judge_native(item: Item, response: Response) -> Judgement:
    return Judgement(answered=not response.abstained, correct=bool((response.detail or {}).get("done")),
                     note=item.meta["job"].about)


register(Task(
    id="computer_use.computerworld_native", area="computer use",
    what="desktop jobs written for this world, graded by the world afterwards",
    dataset=Dataset(name="computerworld desktop jobs (ours)", license="own",
                    url="examples/general_agent/tasks.py", load=_native_items,
                    available=_native_available,
                    fetch_hint="install computerworld (maturin build from its repository)"),
    run=_run_native, judge=_judge_native, controls=("control:abstain",), self_authored=True,
    splits=("dev",),
    notes="self-authored prompts AND grader: a regression suite, never a headline (docs/revival/11)"))
