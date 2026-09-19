"""Subjects: the agent in some configuration, and the controls it must beat.

A number with no floor means nothing, so every task is run against controls as well:

* ``abstain`` never answers — it scores 0 correct and, crucially, 0 wrong. A subject that
  cannot beat it is not useful, and one that is *worse* than it is actively harmful;
* ``echo`` repeats the question — the floor for "did it say anything relevant";
* ``majority`` answers the most common gold answer of the task's training portion, which
  is the classic floor for classification-shaped benchmarks.

The agent subjects differ only in configuration, which is how ablations are run: with the
learned parser or the hand grammar, with or without a plugin.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
from typing import Any, Sequence

from .core import Item, Prompt, Response

ABSTAIN_PHRASES = ("i don't know", "i do not know", "nothing i know", "couldn't recognise",
                   "could not recognise", "didn't fully follow", "did not fully follow", "i can not",
                   "i cannot", "there is nothing there", "it didn't ask me", "i noted",
                   "i read that, but")


def looks_abstained(text: str) -> bool:
    low = text.lower()
    return not text.strip() or any(p in low for p in ABSTAIN_PHRASES)


@dataclass
class Abstainer:
    id: str = "control:abstain"

    def respond(self, prompt: Prompt) -> Response:
        return Response("I don't know.", abstained=True)


@dataclass
class Echo:
    id: str = "control:echo"

    def respond(self, prompt: Prompt) -> Response:
        return Response(prompt.text, abstained=False)


@dataclass
class Majority:
    """Always says the commonest gold answer it was shown (fit on a split, not on test)."""

    answers: Sequence[str] = field(default_factory=tuple)
    id: str = "control:majority"

    @classmethod
    def fit(cls, items: Sequence[Item]) -> "Majority":
        golds = [str(i.gold[0]) if isinstance(i.gold, (list, tuple)) and i.gold else str(i.gold)
                 for i in items if i.gold]
        common = collections.Counter(golds).most_common(1)
        return cls(answers=(common[0][0],) if common else ())

    def respond(self, prompt: Prompt) -> Response:
        return Response(self.answers[0] if self.answers else "", abstained=not self.answers)


@dataclass
class AgentSubject:
    """The tensorcode agent. ``reader`` and ``plugins`` are the knobs an ablation turns."""

    reader: str = "learned"          # "learned" (treebank parser) or "grammar"
    plugins: tuple[str, ...] = ("desktop", "vision")
    id: str = ""
    _agent: Any = None
    _world: Any = None

    def __post_init__(self) -> None:
        self.id = self.id or f"agent:{self.reader}:{'+'.join(self.plugins) or 'none'}"

    def build(self) -> Any:
        from tensorcode.agent import Agent

        reader = None
        if self.reader == "learned":
            from tensorcode.agent.understand import LearnedReader

            reader = LearnedReader()
        plugins = []
        if "desktop" in self.plugins:
            from examples.browser_agents.worlds import desktop_world
            from examples.browser_agents.worlds.runtime import CwWorld
            from examples.general_agent.desktop import DesktopPlugin

            self._world = CwWorld(desktop_world(), 0)
            plugins.append(DesktopPlugin(self._world, learn=False))
        if "vision" in self.plugins:
            from tensorcode.agent.vision_plugin import VisionPlugin

            plugins.append(VisionPlugin())
        return Agent(plugins, reader=reader)

    def fresh(self) -> "AgentSubject":
        """A subject with no memory of earlier items: one world and one store per item."""
        return AgentSubject(reader=self.reader, plugins=self.plugins, id=self.id)

    def respond(self, prompt: Prompt) -> Response:
        if self._agent is None:
            self._agent = self.build()
        for earlier in prompt.history:
            self._agent.turn(earlier[:2000])
        turn = self._agent.turn(prompt.text, images=list(prompt.images))
        return Response(turn.reply, abstained=looks_abstained(turn.reply),
                        detail={"outcomes": [o.status for o in turn.outcomes],
                                "events": [e for e in turn.events if e["type"] in ("act", "receipt", "verified")]})


def named(subject_id: str) -> Any:
    """A subject from its id, so runs can be scripted by name."""
    if subject_id == "control:abstain":
        return Abstainer()
    if subject_id == "control:echo":
        return Echo()
    if subject_id.startswith("agent:"):
        _, reader, plugins = subject_id.split(":", 2)
        return AgentSubject(reader=reader, plugins=tuple(p for p in plugins.split("+") if p != "none"))
    raise ValueError(f"unknown subject: {subject_id}")
