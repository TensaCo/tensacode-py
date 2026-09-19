"""The agent's decisions as tensorcode operations, with implementations to bind.

The agent used to call its own code directly: one reader chosen by a constructor argument,
one planner that scored capabilities inline, one verification that asked the plugin and
believed it. Nothing went through :mod:`tensorcode.ops`, so nothing was traced, nothing
could be substituted, and the library's own machinery — policy, cascade, budget, abstention
— was unreachable from the only program using it.

Four operations carry the turn:

``parse``
    text to a :class:`Transcript`. Two implementations: the treebank-trained reader and the
    hand-written grammar. The learned one *requires* a trained model, so on a host without
    one it is excluded by policy rather than raising, and the grammar answers instead.
``choose``
    which capability to invoke. The agent supplies the objective and the hard constraints;
    :func:`tensorcode.ops.choose` enforces the constraints itself, before any implementation
    sees the options, and an implementation may only return an option that survived them.
``rank``
    which of several remembered answers to say first, by how recently and how confidently
    it was observed.
``verify``
    whether an action had its effect, from a fresh observation rather than from the
    executor's receipt.

Every implementation declares what it needs and what has been measured about it. Where
nothing has been measured the profile stays empty: an unmeasured implementation must not
look fast or good, because a policy that orders by an unknown number is ordering by zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from ..outcomes import Score, Unknown
from ..runtime import Policy, Profile, Request, Runtime, Traits, implementation

#: Where the treebank-trained tagger and parser are cached once ``eval/parsing/train_ud.py``
#: has run. Its absence is a fact about this host, which is what ``Traits.requires`` is for.
MODEL = Path.home() / ".cache" / "tensorcode" / "models" / "ud_ewt_parser.pickle"


@dataclass(frozen=True)
class Transcript:
    """One message, read: its sentences with their acts, and which reader produced them.

    A distinct type from a bare list so ``parse`` can validate what it got back, and so the
    reader that produced a reading travels with it into the trace.
    """

    sentences: tuple[Any, ...] = ()
    by: str = ""

    def __iter__(self):
        return iter(self.sentences)

    def __len__(self) -> int:
        return len(self.sentences)


# ------------------------------------------------------------------ reading


_LEARNED: Any = None


def learned_reader() -> Any:
    """The treebank reader, built once per process (it loads a model and a lemma table)."""
    global _LEARNED
    if _LEARNED is None:
        from .understand import LearnedReader

        _LEARNED = LearnedReader()
    return _LEARNED


def install_learned_reader(reader: Any) -> None:
    """Reuse a reader the caller already built, instead of loading the model twice."""
    global _LEARNED
    _LEARNED = reader


def _wants(name: str):
    """Accept a parse request unless it asked for a different reader by name."""

    def accepts(request: Request) -> bool:
        return request.target is Transcript and request.params.get("prefer") in (None, name)

    return accepts


@implementation(
    "parse",
    name="reader:grammar",
    version="1",
    accepts=_wants("grammar"),
    traits=Traits(locality="in_process", deterministic=True),
    # coverage and act accuracy are measured per task by the assay, not per call here
    profile=Profile(source="eval/results/assay.jsonl"),
)
def _read_with_grammar(request: Request) -> Any:
    from ..language import ENGLISH
    from .understand import read

    grammar = request.params.get("grammar") or ENGLISH
    return Transcript(tuple(read(grammar, request.subject)), "reader:grammar")


@implementation(
    "parse",
    name="reader:learned",
    version="1",
    accepts=_wants("learned"),
    traits=Traits(locality="in_process", deterministic=True, requires=frozenset({"ud-parser"})),
    profile=Profile(source="eval/results/parsing_ud.json",
                    quality={"las": 0.7696, "uas": 0.8248, "tagging": 0.9359}),
)
def _read_with_treebank(request: Request) -> Any:
    try:
        reader = learned_reader()
    except FileNotFoundError as exc:
        # abstention, not failure: the next implementation in the cascade should read it
        return Unknown("no_model", str(exc))
    return Transcript(tuple(reader.read(request.subject)), "reader:learned")


# ------------------------------------------------------------------ choosing


@implementation(
    "choose",
    name="chooser:utility",
    version="1",
    traits=Traits(locality="in_process", deterministic=True),
    profile=Profile(),
)
def _choose_by_utility(request: Request) -> Any:
    """The feasible option the objective scores highest.

    The objective comes from the caller, so what "better" means is the agent's to say and
    this implementation's only job is the argmax. An objective that cannot score its options
    is an abstention, never an arbitrary pick.
    """
    objective = request.target
    options: Sequence[Any] = request.subject
    if objective is None or objective.utility is None:
        return Unknown("no_utility", f"objective {getattr(objective, 'name', '?')} scores nothing")
    scored = [(objective.utility(option, request.params.get("given")), option) for option in options]
    if not scored:
        return Unknown("no_options", "nothing feasible was proposed")
    best = max(scored, key=lambda pair: pair[0])
    return best[1]


# ------------------------------------------------------------------ ranking


@implementation(
    "rank",
    name="ranker:evidence",
    version="1",
    traits=Traits(locality="in_process", deterministic=True),
    profile=Profile(),
)
def _rank_by_evidence(request: Request) -> Any:
    """Order remembered answers by how recently and how confidently they were observed.

    Candidates are ``(value, record)`` pairs. Recency first: what was observed later is
    what the store was told later, and a claim whose source stated no confidence is not
    treated as certain — it ranks below one that stated a high confidence and above one
    that stated a low one.
    """
    candidates = request.params.get("candidates") or ()
    if not all(isinstance(c, tuple) and len(c) == 2 for c in candidates):
        return Unknown("unrankable", "candidates are not (value, record) pairs")
    order = []
    for pair in candidates:
        value, record = pair
        evidence = getattr(record, "evidence", ()) or ()
        latest = max((e.observed_at for e in evidence if e.observed_at is not None),
                     default=datetime.min.replace(tzinfo=timezone.utc))
        stated = [e.confidence.value for e in evidence if e.confidence is not None]
        order.append((latest, max(stated) if stated else 0.5, pair))
    order.sort(key=lambda row: (row[0], row[1]), reverse=True)
    n = len(order)
    # a ranking returns the candidates it was given, in order, so the caller keeps whatever
    # else it attached to them
    return [(pair, Score(1.0 - i / max(n, 1), "relevance")) for i, (_, _, pair) in enumerate(order)]


# ------------------------------------------------------------------ the runtime


def host_capabilities() -> frozenset[str]:
    """What this machine actually has. Nothing is assumed present."""
    found = set()
    if MODEL.exists():
        found.add("ud-parser")
    try:
        import sklearn  # noqa: F401

        found.add("sklearn")
    except Exception:  # noqa: BLE001
        pass
    return frozenset(found)


IMPLEMENTATIONS = (_read_with_grammar, _read_with_treebank, _choose_by_utility, _rank_by_evidence)


def agent_runtime(*, prefer_reader: str | None = None, policy: Policy | None = None) -> Runtime:
    """A runtime with the agent's operations bound.

    Declared order puts the hand grammar first, which is what an agent built with no
    preference has always used. Naming a reader, or handing in a policy that orders by a
    measured profile, is how that is changed — the choice is the runtime's, not a branch
    inside the agent.
    """
    order = list(IMPLEMENTATIONS)
    if prefer_reader == "learned":
        order = [_read_with_treebank, _read_with_grammar, _choose_by_utility, _rank_by_evidence]
    return Runtime(order, policy=policy or Policy(available=host_capabilities()))


@dataclass
class Plan:
    """One way to satisfy a request: a plugin's capability with arguments that fit it."""

    plugin: Any
    capability: Any
    args: dict[str, Any] = field(default_factory=dict)
    met: int = 0                       # goal conditions this capability's effects achieve
    achieves_all_specified: bool = True  # every fully-specified condition is among the effects
    fully_applied: bool = True         # every parameter of the capability has an argument

    def describe(self) -> str:
        return f"{self.plugin.name}.{self.capability.name}({', '.join(sorted(self.args))})"

    def __repr__(self) -> str:  # what the trace shows
        return self.describe()
