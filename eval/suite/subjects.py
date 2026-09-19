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

#: The outcome statuses that mean the agent committed to something: it answered a question,
#: or it carried out a request. Everything else — unknown, declined, not_understood, noted,
#: failed, unverified, mentioned — is the agent saying it did not.
COMMITTED = frozenset({"answered", "done"})

ABSTAIN_PHRASES = ("i don't know", "i do not know", "nothing i know", "couldn't recognise",
                   "could not recognise", "didn't fully follow", "did not fully follow", "i can not",
                   "i cannot", "there is nothing there", "it didn't ask me", "i noted",
                   "i read that, but")


def looks_abstained(text: str) -> bool:
    """A last resort for a subject that reports nothing structured about itself.

    The agent no longer needs this — it says what each act came to — but a plain text
    subject (a control, or something wrapped from outside) has only its words.
    """
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
            plugins.append(DesktopPlugin(self._world, learned=learned_capabilities(self._world)))
        if "vision" in self.plugins:
            from tensorcode.agent.vision_plugin import VisionPlugin

            plugins.append(VisionPlugin())
        return Agent(plugins, reader=reader)

    def unavailable(self) -> str:
        """Why this subject cannot be built here, or "" if it can.

        Asked once before a run rather than discovered per item: a subject whose plugin is
        missing raises on every item, and a column of crashes is easy to mistake for a
        column of results.
        """
        try:
            self.build()
        except Exception as exc:  # noqa: BLE001
            return f"{type(exc).__name__}: {exc}"
        return ""

    def fresh(self) -> "AgentSubject":
        """A subject with no memory of earlier items: one world and one store per item."""
        return AgentSubject(reader=self.reader, plugins=self.plugins, id=self.id)

    def respond(self, prompt: Prompt) -> Response:
        if self._agent is None:
            self._agent = self.build()
        for earlier in prompt.history:
            self._agent.turn(earlier[:2000])
        turn = self._agent.turn(prompt.text, images=list(prompt.images))
        # the agent's own account of what it did, rather than a search of its prose for
        # phrases like "i don't know": a decline the grammar could not realize was scored as
        # a confident wrong answer, three times over, on the task about *asking* instead of
        # answering
        committed = any(o.status in COMMITTED for o in turn.outcomes)
        return Response(turn.reply, abstained=not committed,
                        detail={"outcomes": [o.status for o in turn.outcomes],
                                "events": [e for e in turn.events if e["type"] in ("act", "receipt", "verified")]})


_LEARNED: list | None = None


def learned_capabilities(world: Any) -> list:
    """What commands on this desktop do, discovered once and reused across items.

    The subject used to be built with ``learn=False``, which left the plugin with no
    capabilities at all: the desktop agent was measured on tasks it had no means to attempt,
    and scored as though it had declined them. Discovery is an experiment against a snapshot
    (about sixteen seconds), and what it finds is a fact about this *kind* of machine, so it
    is done once per process rather than once per item.
    """
    global _LEARNED
    if _LEARNED is None:
        from examples.general_agent.desktop import DesktopPlugin
        from examples.general_agent.discover import CANDIDATES, discover

        probe = DesktopPlugin(world, learn=False)
        _LEARNED = discover(probe, world, CANDIDATES)
    return _LEARNED


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
