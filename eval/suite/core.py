"""The assay's vocabulary: subjects, tasks, items, judgements, results.

One idea per type, so a new benchmark is a new :class:`Task` and nothing else changes:

* a :class:`Subject` is a thing under test — the agent in some configuration, or a
  control (always abstain, majority answer). Everything is measured against controls,
  because a number without a floor says nothing;
* a :class:`Task` is a dataset plus a way to run it and a way to grade it. It declares
  where its data comes from, under what licence, and whether the data is present;
* a :class:`Judgement` separates **abstained** from **answered**, and answered into
  **correct** or **wrong**. The whole design rests on that distinction, so the metric
  layer keeps it rather than collapsing to accuracy;
* a :class:`Result` is one (task, subject, split) run, with the provenance needed to
  believe it: the data's hash, the code's commit, the time.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence, runtime_checkable


@dataclass(frozen=True)
class Prompt:
    """What a subject is given: words, pictures, and any earlier turns."""

    text: str
    images: tuple[bytes, ...] = ()
    history: tuple[str, ...] = ()
    context: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Response:
    """What a subject gives back. ``abstained`` is the subject's own refusal to answer."""

    text: str
    abstained: bool = False
    detail: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class Subject(Protocol):
    id: str

    def respond(self, prompt: Prompt) -> Response: ...


@dataclass(frozen=True)
class Item:
    id: str
    prompt: Prompt
    gold: Any = None
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Judgement:
    answered: bool
    correct: bool | None = None       # None when the item has no gold answer
    score: float | None = None        # for tasks scored continuously (attachment, F1)
    note: str = ""


@dataclass(frozen=True)
class Dataset:
    """Where a task's items come from, and whether they are here."""

    name: str
    license: str
    url: str
    load: Callable[[str], Sequence[Item]]        # split -> items
    available: Callable[[], bool] = lambda: True
    fetch_hint: str = ""

    def status(self) -> str:
        return "ready" if self.available() else "data missing"


@dataclass(frozen=True)
class Task:
    """A benchmark: its data, how a subject is run on it, and how the result is judged."""

    id: str
    area: str                                     # language | knowledge | reasoning | memory | vision | tools | ...
    what: str                                     # one line, in plain words
    dataset: Dataset
    judge: Callable[[Item, Response], Judgement]
    splits: tuple[str, ...] = ("dev",)
    headline: str = "accuracy"                    # which metric the scorecard shows
    run: Callable[[Subject, Item], Response] | None = None   # default: subject.respond
    controls: tuple[str, ...] = ()                # subject ids that must also be run
    self_authored: bool = False                   # True when we wrote the data or the world
    notes: str = ""

    def items(self, split: str) -> Sequence[Item]:
        return self.dataset.load(split)

    def execute(self, subject: Subject, item: Item) -> Response:
        return self.run(subject, item) if self.run else subject.respond(item.prompt)


_REGISTRY: dict[str, Task] = {}


def register(task: Task) -> Task:
    if task.id in _REGISTRY:
        raise ValueError(f"duplicate task id: {task.id}")
    _REGISTRY[task.id] = task
    return task


def registry() -> Mapping[str, Task]:
    if not _REGISTRY:
        load_tasks()
    return dict(_REGISTRY)


def load_tasks() -> None:
    """Import every task module, so registering is a side effect of being in the package."""
    import importlib
    import pkgutil

    from . import tasks as tasks_package

    for module in pkgutil.iter_modules(tasks_package.__path__):
        importlib.import_module(f"{tasks_package.__name__}.{module.name}")


def metrics(judgements: Iterable[Judgement]) -> dict:
    """Answered / correct / wrong kept apart, plus the scores of continuous tasks."""
    js = list(judgements)
    answered = [j for j in js if j.answered]
    gradable = [j for j in js if j.correct is not None]
    correct = [j for j in gradable if j.correct]
    scored = [j.score for j in js if j.score is not None]
    out = {
        "n": len(js),
        "answered": len(answered),
        "abstained": len(js) - len(answered),
        "correct": len(correct),
        "wrong": len([j for j in gradable if j.answered and not j.correct]),
        "accuracy": round(len(correct) / len(gradable), 4) if gradable else None,
        "precision_when_answering": round(len(correct) / len(answered), 4) if answered and gradable else None,
        "coverage": round(len(answered) / len(js), 4) if js else None,
    }
    if scored:
        out["score"] = round(sum(scored) / len(scored), 4)
    return out


def commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=Path(__file__).parent, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"
