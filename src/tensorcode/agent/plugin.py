"""What a plugin gives the agent: words, kinds of things, and what its actions achieve.

A plugin never sees the user's text and the agent never sees a plugin's internals.
They meet in one vocabulary:

* **effects** are VerbNet result predicates over thematic roles (``be(Result)``,
  ``has_location(Theme, Destination)``, ``has_state(Patient)``, ``destroyed(Patient)``)
  — the same predicates a parsed request becomes (``language/verbnet.py``);
* **kinds** are nouns: a parameter wants a ``directory``, and a description fills it
  if its noun is a kind of ``directory`` (WordNet's hierarchy plus the plugin's own
  links, e.g. "folder is a directory" on a desktop, which WordNet does not know);
* **informs** says which predicates an action reveals about the world, which is how
  a question ("what's on my desktop?") finds an action that can answer it.

The agent chooses which capability to run by matching effects and kinds. A plugin
cannot route a phrase to an action, because it is never given a phrase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from ..actions import action
from ..outcomes import Receipt, Unknown
from ..records import Claim


@dataclass(frozen=True)
class Param:
    name: str
    kind: str                      # a noun: what may fill it
    role: str | None = None        # the thematic role it realizes, if any


@dataclass(frozen=True)
class Effect:
    """``pred(role=param, ...)`` holds after the capability runs (or stops holding, if negated)."""

    pred: str
    roles: Mapping[str, str]       # thematic role -> parameter name
    negated: bool = False


@dataclass(frozen=True)
class Informs:
    """Running the capability reveals every true ``pred(...)`` with ``role`` bound to ``param``."""

    pred: str
    role: str
    param: str


@dataclass(frozen=True)
class Capability:
    name: str
    params: tuple[Param, ...]
    effects: tuple[Effect, ...] = ()
    informs: tuple[Informs, ...] = ()
    effect_kind: str = "write"     # "read" | "write" | "external" (tensorcode.actions)
    description: str = ""          # for people reading traces, never matched against text

    def param(self, name: str) -> Param | None:
        return next((p for p in self.params if p.name == name), None)


@action(effect="write", idempotent=False)
@dataclass(frozen=True)
class Call:
    """One capability invocation, as a registered action so it gets a receipt and a trace span."""

    plugin: str
    capability: str
    args: tuple[tuple[str, Any], ...]

    def arg(self, name: str) -> Any:
        return dict(self.args).get(name)


@dataclass
class Plugin:
    """Base class. Subclasses override the methods; the fields are data."""

    name: str
    lexicon: tuple = ()                                        # tensorcode.language Entry values
    kinds: Mapping[str, tuple[str, ...]] = field(default_factory=dict)  # noun -> nouns it is a kind of here

    def capabilities(self) -> Sequence[Capability]:
        return ()

    def perceive(self) -> Iterable[Claim]:
        """What is true now, as claims. Called before answering and after acting."""
        return ()

    def refer(self, description: Any, param: Param, *, context: Mapping[str, Any]) -> Any | Unknown:
        """The world entity (a ``Ref``) a description denotes when it fills ``param``.

        For a parameter naming something to be *made*, the entity may not exist yet
        (the folder "recipes on my desktop" is where it will be). Unknown when the
        description picks out nothing, or more than one thing.
        """
        return Unknown("cannot_refer", f"{self.name} cannot resolve {description!r} as {param.kind}")

    def display(self, ref: Any) -> str:
        """How to name a world entity to a person (a file's name, not its full path)."""
        return getattr(ref, "id", str(ref)).split(":", 1)[-1]

    def denote(self, description: Any) -> Any | Unknown:
        """The existing world entity a description picks out, for answering questions."""
        return Unknown("cannot_refer", f"{self.name} does not know {description!r}")

    def execute(self, act: Call, *, key: str | None) -> Receipt:
        return Receipt(act, "rejected", error=f"{self.name} does not implement {act.capability}")

    def holds(self, cap: Capability, args: Mapping[str, Any]) -> bool | Unknown:
        """Whether ``cap``'s effects hold now, judged from a fresh observation."""
        return Unknown("no_check", f"{self.name} cannot check {cap.name}")

    def reveal(self, cap: Capability, args: Mapping[str, Any], receipt: Receipt) -> Iterable[Claim]:
        """Claims learned by running an informing capability (``Informs``)."""
        return ()


def describe_capabilities(plugins: Iterable[Plugin]) -> list[dict]:
    """A plain listing, for the viewer and for "what can you do?" answered from data."""
    out = []
    for p in plugins:
        for c in p.capabilities():
            out.append({"plugin": p.name, "name": c.name, "params": [(x.name, x.kind) for x in c.params],
                        "effects": [("not " if e.negated else "") + f"{e.pred}({', '.join(f'{r}={v}' for r, v in e.roles.items())})" for e in c.effects],
                        "informs": [f"{i.pred}({i.role}={i.param})" for i in c.informs]})
    return out

