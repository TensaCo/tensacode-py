"""Acting: computer use, safety, and tool choice."""

from __future__ import annotations

from ..core import Dataset, Task, register
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

register(Task(
    id="computer_use.computerworld_native", area="computer use",
    what="tasks written for computerworld itself, graded by world state",
    dataset=_missing("computerworld task set (ours)", "own", "examples/browser_agents/tasks",
                     "write tasks + world-state graders; self-authored, so a regression suite, not a headline"),
    judge=attempted_only, self_authored=True,
    notes="self-authored environment AND grader: never a headline number (docs/revival/11)"))
