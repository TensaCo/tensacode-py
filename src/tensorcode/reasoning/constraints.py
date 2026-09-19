"""Search over candidate worlds, so that a stated constraint can rule one out.

The owner's probe, verbatim: three boxes labelled a, b and c, exactly one holding a prize;
each box carries a statement — a: "the prize is not in this box", b: "the prize is in box
a", c: "the prize is not in box b" — and **exactly one of the three statements is true**.
The agent answered "I don't know" and "I can not explain my reasoning".

It was not a reading failure. The propositions come out of that text fine, and the store
will happily hold all three candidate locations side by side as ``hypothesised``. What was
missing is the step *after* holding them: nothing in the library could take a finite set of
mutually exclusive possibilities and ask, of each one, **how many of these other
propositions would then be true** — and drop the ones where the count is wrong. Retrieval
answers "what is on record"; a puzzle asks "which of these *could* be on record, given what
the text says must come out true". That is a search, and its answer is worth nothing without
the eliminations, because "box b" with no reasoning is indistinguishable from a guess.

Three parts:

:class:`Variable`
    one open question, with the finite set of hypotheses that could settle it. A variable's
    domain is *exhaustive and mutually exclusive* by construction — which is exactly what
    "exactly one box contains the prize" states — so choosing one member makes the others
    false, and that is where nearly all the deductive force comes from.
:class:`Constraint`
    anything that can judge a whole world. :class:`MustHold` is a plain stated fact;
    :class:`TruthCount` is the crux, a constraint *over the truth of other propositions*
    ("exactly one of these three is true"), which no per-option predicate can express;
    :class:`Distinct` relates the fillers two questions settle on.
:func:`solve`
    enumerates, eliminates, and reports both the survivors and which constraint ruled out
    each candidate.

Two rules it will not break, both learned elsewhere in this library:

* **Unknown is not false.** A proposition no variable ranges over is undetermined, not
  false, and a constraint that cannot be decided does *not* eliminate a world — it puts it
  in :attr:`Solution.undecided`. Eliminating on an undetermined constraint would let the
  solver report a unique answer it never earned, which is the same wireheading as a phrase
  list that makes a decline look like an answer (docs/revival/32).
* **Ambiguity is reported, not resolved.** If two worlds survive, :meth:`Solution.answer`
  returns :class:`Unknown` naming both. Picking one would be a coin flip wearing a proof.

Note the name: ``ops.Constraint`` tests one *option* against a given, before a backend sees
the options; a constraint here tests one *world*. They are different animals, and only this
one can talk about the truth of another proposition.
"""

from __future__ import annotations

import dataclasses
import itertools
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Iterator, Protocol, runtime_checkable

from ..outcomes import Unknown, Verdict
from ..records import Proposition
from ..runtime import Profile, Request, Traits, implementation

#: A proposition reduced to what it says about the world: its content, and whether it
#: affirms or denies it. Two propositions with the same literal are the same thing said.
Literal = tuple[str, bool]


def _negate(literal: Literal) -> Literal:
    return (literal[0], not literal[1])


# ------------------------------------------------------------------ open questions


@dataclass(frozen=True)
class Variable:
    """One open question and the hypotheses that could settle it; exactly one of them holds.

    This is not :class:`~tensorcode.records.Var`, which is a hole in a retrieval pattern.
    A hole asks the store what it already knows; a variable here asks what the store *would*
    know under each of several readings of the world, none of which is believed.

    The domain must be exhaustive and mutually exclusive, because that is what the solver
    reasons from: "the prize is not in this box" is decidable only because choosing box b
    makes ``contains(box:a, prize)`` *false* rather than merely unsupported. When a puzzle
    does not say its alternatives are exhaustive, add the possibility it leaves open (an
    ``elsewhere`` filler) instead of quietly letting the solver assume it away.
    """

    name: str
    domain: tuple[Proposition, ...]

    def __post_init__(self) -> None:
        if not self.domain:
            raise ValueError(f"question {self.name!r} has no hypotheses; nothing to choose between")
        seen: dict[Literal, Proposition] = {}
        for hypothesis in self.domain:
            key = _literal_of(hypothesis)
            if key in seen:
                raise ValueError(f"question {self.name!r} lists {hypothesis.describe()!r} twice")
            seen[key] = hypothesis

    def __len__(self) -> int:
        return len(self.domain)


def one_of(name: str, *hypotheses: Proposition) -> Variable:
    """A question over alternatives, each stamped ``hypothesised``.

    The modality is not decoration: a candidate location for the prize must be storable
    beside its rivals without any of them being believed, and ``hypothesised`` is the
    substrate's word for that (records.MODALITIES). Stamping it here means a caller cannot
    accidentally build a domain out of propositions the store would read as asserted facts.
    """
    return Variable(name, tuple(_as_hypothesis(h) for h in hypotheses))


def _as_hypothesis(proposition: Proposition) -> Proposition:
    if proposition.modality == "hypothesised":
        return proposition
    if proposition.modality != "asserted":
        raise ValueError(
            f"a hypothesis must be asserted or hypothesised content, got {proposition.modality!r}: "
            f"{proposition.describe()} — a desire or an obligation is not a candidate for being true"
        )
    return dataclasses.replace(proposition, modality="hypothesised")


def _literal_of(proposition: Proposition) -> Literal:
    """The proposition's content identity, with ``hypothesised`` folded into ``asserted``.

    Folding only that one modality is deliberate. A hypothesis *is* a candidate for being
    true, so "hypothesised: the prize is in a" and "the prize is in a" have to be the same
    thing said or no statement would ever match a candidate world. ``desired: X`` and ``X``
    are not the same thing said, and collapsing them would make the solver prove that
    whatever anyone wants is the case.
    """
    normal = proposition
    if not proposition.polarity or proposition.modality == "hypothesised":
        normal = dataclasses.replace(
            proposition,
            polarity=True,
            modality="asserted" if proposition.modality == "hypothesised" else proposition.modality,
        )
    return normal.id, proposition.polarity


def _plain(proposition: Proposition) -> str:
    """How a proposition reads in a trace: its content, without the hypothesis marker.

    Every line of an elimination trace is about a candidate, so printing
    ``hypothesised:`` on all of them is noise that hides the part a reader is comparing.
    """
    if proposition.modality == "hypothesised":
        proposition = dataclasses.replace(proposition, modality="asserted")
    return proposition.describe()


# --------------------------------------------------------------------- candidates


@dataclass(frozen=True)
class World:
    """One candidate assignment: what every open question would be, if this were the world.

    ``puzzle`` travels with the assignment because the truth of a proposition is not a
    property of the assignment alone — it also needs the domains, which say what choosing
    one alternative makes false. It is excluded from equality so two worlds compare by their
    choices, as a reader would expect.
    """

    puzzle: Puzzle = field(compare=False, repr=False)
    choices: tuple[tuple[str, Proposition], ...] = ()

    def __getitem__(self, name: str) -> Proposition:
        for chosen, hypothesis in self.choices:
            if chosen == name:
                return hypothesis
        raise KeyError(name)

    def get(self, name: str) -> Proposition | None:
        try:
            return self[name]
        except KeyError:
            return None

    @cached_property
    def literals(self) -> frozenset[Literal]:
        return frozenset(_literal_of(h) for _, h in self.choices)

    def truth(self, proposition: Proposition) -> bool | None:
        """Is ``proposition`` true in this world? ``None`` means the world does not say."""
        return self.puzzle.truth(self, proposition)

    def describe(self) -> str:
        return "; ".join(f"{name} = {_plain(h)}" for name, h in self.choices)


# -------------------------------------------------------------------- constraints


@runtime_checkable
class Constraint(Protocol):
    """Something a world either satisfies, violates, or leaves undecided.

    A protocol rather than a closed set, so a caller can add a constraint shape this module
    did not anticipate without it becoming a callable escape hatch: a constraint has to be
    able to *say why*, and a lambda cannot.
    """

    @property
    def name(self) -> str: ...

    def judge(self, world: World) -> Verdict: ...


@dataclass(frozen=True)
class MustHold:
    """A proposition the puzzle states outright, in a world where it must come out true.

    Polarity does the work for denials: ``MustHold(Proposition(..., polarity=False))`` is
    "the prize is not in box a", and it eliminates exactly the world that puts it there.
    """

    proposition: Proposition
    label: str = ""

    @property
    def name(self) -> str:
        return self.label or f"must hold: {_plain(self.proposition)}"

    def judge(self, world: World) -> Verdict:
        truth = world.truth(self.proposition)
        if truth is None:
            return Verdict("unknown", (f"nothing in this world settles {_plain(self.proposition)}",))
        if truth:
            return Verdict("holds", (f"{_plain(self.proposition)} is true here",))
        return Verdict("fails", (f"{_plain(self.proposition)} is false here",))


@dataclass(frozen=True)
class TruthCount:
    """How many of these propositions are true — the constraint a puzzle turns on.

    "Exactly one of these three statements is true" is not a property of any one candidate;
    it is a property of a world, obtained by evaluating three *other* propositions in it and
    counting. This is the one shape that cannot be pushed down into a per-option predicate,
    which is why the agent could not do stated-constraint puzzles at all: it had `check` for
    one claim and `choose` for one option, and nothing that quantified over truth.

    Counting under partial information is done as an interval, not a guess. With ``k``
    members known true and ``u`` undetermined, the true count lies in ``[k, k+u]``; the
    constraint fails only when every value in that interval is excluded, and holds only when
    every value in it is allowed. So an undetermined member weakens the conclusion instead
    of silently counting as false — which would eliminate worlds on the strength of an
    unasked question.
    """

    members: tuple[Proposition, ...]
    exactly: int | None = None
    at_least: int | None = None
    at_most: int | None = None
    label: str = ""

    def __post_init__(self) -> None:
        if not self.members:
            raise ValueError("a truth count over no propositions constrains nothing")
        if self.exactly is None and self.at_least is None and self.at_most is None:
            raise ValueError("a truth count needs a bound: exactly, at_least, or at_most")

    @property
    def name(self) -> str:
        if self.label:
            return self.label
        bounds = [
            f"exactly {self.exactly}" if self.exactly is not None else "",
            f"at least {self.at_least}" if self.at_least is not None else "",
            f"at most {self.at_most}" if self.at_most is not None else "",
        ]
        return f"{' and '.join(b for b in bounds if b)} of {len(self.members)} propositions true"

    def judge(self, world: World) -> Verdict:
        true_here, false_here, undetermined = [], [], []
        for member in self.members:
            truth = world.truth(member)
            (true_here if truth else undetermined if truth is None else false_here).append(_plain(member))
        low, high = len(true_here), len(true_here) + len(undetermined)
        detail = [f"true: {'; '.join(true_here) or '—'}", f"false: {'; '.join(false_here) or '—'}"]
        if undetermined:
            detail.append(f"undetermined: {'; '.join(undetermined)}")
        count = f"{low} true" if low == high else f"between {low} and {high} true"
        verdicts = []
        if self.exactly is not None:
            verdicts.append(_within(low, high, self.exactly, self.exactly))
        if self.at_least is not None:
            verdicts.append(_within(low, high, self.at_least, None))
        if self.at_most is not None:
            verdicts.append(_within(low, high, None, self.at_most))
        if "fails" in verdicts:
            return Verdict("fails", (f"{count}, which this constraint forbids", *detail))
        if all(v == "holds" for v in verdicts):
            return Verdict("holds", (f"{count}", *detail))
        return Verdict("unknown", (f"{count}; not enough is settled to tell", *detail))


def exactly_one(*members: Proposition, label: str = "") -> TruthCount:
    """"Exactly one of these is true" — the commonest stated constraint, spelled out once."""
    return TruthCount(members, exactly=1, label=label)


def _within(low: int, high: int, floor: int | None, ceiling: int | None) -> str:
    """Compare the interval ``[low, high]`` of possible counts against one bound."""
    if ceiling is not None and low > ceiling:
        return "fails"
    if floor is not None and high < floor:
        return "fails"
    if (floor is None or low >= floor) and (ceiling is None or high <= ceiling):
        return "holds"
    return "unknown"


@dataclass(frozen=True)
class Distinct:
    """Two or more questions must not settle on the same filler for a role.

    "The key is not in the box with the prize" is a relation *between* answers, so it cannot
    be a fact about either one. Comparing role fillers keeps it structural: no callable, and
    the reason it gives names the two questions that collided.
    """

    variables: tuple[str, ...]
    role: str
    label: str = ""

    def __post_init__(self) -> None:
        if len(self.variables) < 2:
            raise ValueError("distinctness needs at least two questions to compare")

    @property
    def name(self) -> str:
        return self.label or f"distinct {self.role} across {', '.join(self.variables)}"

    def judge(self, world: World) -> Verdict:
        seen: dict[Any, str] = {}
        for name in self.variables:
            hypothesis = world.get(name)
            if hypothesis is None:
                return Verdict("unknown", (f"this world has no question called {name!r}",))
            if self.role not in hypothesis.roles:
                return Verdict("unknown", (f"{name} = {_plain(hypothesis)} fills no {self.role!r} role",))
            filler = hypothesis.roles[self.role]
            if filler in seen:
                return Verdict("fails", (f"{name} and {seen[filler]} share {self.role}={filler}",))
            seen[filler] = name
        return Verdict("holds", (f"{self.role} differs across {', '.join(self.variables)}",))


# ------------------------------------------------------------------------ puzzles


@dataclass(frozen=True)
class Puzzle:
    """The finite space to search: the open questions, what must come out true, what is settled.

    ``given`` are propositions true in every world — the part of the situation that is not in
    question. They are kept separate from the constraints because a fact that holds
    everywhere cannot eliminate anything, and pretending otherwise fills a trace with
    constraints that never fire.
    """

    variables: tuple[Variable, ...]
    constraints: tuple[Constraint, ...] = ()
    given: tuple[Proposition, ...] = ()

    def __post_init__(self) -> None:
        if not self.variables:
            raise ValueError("a puzzle with no open question has nothing to search")
        owner: dict[Literal, str] = {}
        for variable in self.variables:
            for hypothesis in variable.domain:
                key = _literal_of(hypothesis)
                for candidate in (key, _negate(key)):
                    if candidate in owner and owner[candidate] != variable.name:
                        raise ValueError(
                            f"questions {owner[candidate]!r} and {variable.name!r} both range over "
                            f"{_plain(hypothesis)}; whichever answered first would overrule the other"
                        )
                owner[key] = variable.name

    @cached_property
    def size(self) -> int:
        """How many candidate worlds there are: the product of the domain sizes."""
        total = 1
        for variable in self.variables:
            total *= len(variable)
        return total

    @cached_property
    def _by_domain(self) -> dict[Literal, str]:
        return {_literal_of(h): v.name for v in self.variables for h in v.domain}

    @cached_property
    def _given_literals(self) -> frozenset[Literal]:
        return frozenset(_literal_of(g) for g in self.given)

    @cached_property
    def _atoms(self) -> dict[str, Literal]:
        """Memo from a proposition's id to its literal, so a long search hashes each once."""
        return {}

    def literal(self, proposition: Proposition) -> Literal:
        key = self._atoms.get(proposition.id)
        if key is None:
            key = self._atoms[proposition.id] = _literal_of(proposition)
        return key

    def truth(self, world: World, proposition: Proposition) -> bool | None:
        """Is ``proposition`` true in ``world``? ``None`` means this puzzle does not say.

        The order matters. What the world chose comes first; then what was given; then
        exclusivity — a hypothesis its own question passed over is *false*, and so the
        denial of one is *true*, which is how "the prize is not in this box" gets a truth
        value at all. Anything else is undetermined, including a proposition about a box no
        question ranges over: a domain being exhaustive over its own alternatives says
        nothing about a fourth box nobody mentioned, and answering "false" there would be an
        assumption dressed as a deduction.
        """
        key = self.literal(proposition)
        if key in world.literals:
            return True
        if _negate(key) in world.literals:
            return False
        if key in self._given_literals:
            return True
        if _negate(key) in self._given_literals:
            return False
        if key in self._by_domain:
            return False  # its question was settled otherwise, and a question has one answer
        if _negate(key) in self._by_domain:
            return True  # what it denies was passed over, so the denial holds
        return None

    def worlds(self) -> Iterator[World]:
        """Every candidate, in declared order, so a trace reads the same way twice."""
        names = tuple(v.name for v in self.variables)
        for combination in itertools.product(*(v.domain for v in self.variables)):
            yield World(self, tuple(zip(names, combination)))


# ------------------------------------------------------------------------- solving

#: Worlds this solver will enumerate without being told to. Enumeration is
#: ``O(worlds x constraint members)`` — the product of the domain sizes times the work of
#: judging one world — so the cap is on the product, checked from the domain sizes alone
#: before anything is enumerated. Refusing in O(number of questions) is the point: a puzzle
#: encoded with one variable too many should say so in a millisecond, not appear to be
#: thinking for an hour. Raise it deliberately, with a number you are willing to wait for.
MAX_WORLDS = 100_000


@dataclass(frozen=True)
class Elimination:
    """One candidate, and the single constraint that decided its fate.

    The first failing constraint is recorded and judging that world stops there. A reader
    asking "why not box a?" wants the reason, not all the reasons, and the short-circuit is
    also what keeps a large search cheap.
    """

    world: World
    constraint: str
    reasons: tuple[str, ...]

    def describe(self) -> str:
        detail = "".join(f"\n      {r}" for r in self.reasons)
        return f"{self.world.describe()}\n    by {self.constraint}{detail}"


@dataclass(frozen=True)
class Solution:
    """What survived, what did not, and why — a result without the eliminations is a guess."""

    puzzle: Puzzle
    surviving: tuple[World, ...]
    eliminated: tuple[Elimination, ...]
    undecided: tuple[Elimination, ...] = ()

    @property
    def considered(self) -> int:
        return len(self.surviving) + len(self.eliminated) + len(self.undecided)

    @property
    def unique(self) -> bool:
        """One world survived *and* every other was actually ruled out."""
        return len(self.surviving) == 1 and not self.undecided

    def answer(self) -> World | Unknown:
        """The one world that works, or an :class:`Unknown` that says why there isn't one.

        An undecided candidate blocks a unique answer even when exactly one world survives,
        because a world that was never ruled out is still a rival. Reporting "box b" while
        holding a candidate we could not judge would be a claim about reasoning we did not do.
        """
        if self.undecided:
            blocked = "; ".join(f"{e.world.describe()} ({e.constraint} undecided)" for e in self.undecided)
            return Unknown("undetermined", f"{len(self.undecided)} candidate(s) could not be judged: {blocked}")
        if not self.surviving:
            return Unknown("no_world", f"every one of {self.considered} candidates was ruled out")
        if len(self.surviving) > 1:
            worlds = "; ".join(w.describe() for w in self.surviving)
            return Unknown("ambiguous", f"{len(self.surviving)} worlds satisfy every constraint: {worlds}")
        return self.surviving[0]

    def explain(self) -> str:
        """The elimination trace, as prose a person can check line by line."""
        questions = len(self.puzzle.variables)
        lines = [
            f"{self.considered} candidate worlds from {questions} open question"
            f"{'' if questions == 1 else 's'}: {len(self.surviving)} surviving, "
            f"{len(self.eliminated)} ruled out"
            + (f", {len(self.undecided)} undecided" if self.undecided else "")
        ]
        for elimination in self.eliminated:
            lines.append(f"  ruled out {elimination.describe()}")
        for elimination in self.undecided:
            lines.append(f"  undecided {elimination.describe()}")
        for world in self.surviving:
            lines.append(f"  survives  {world.describe()}")
        return "\n".join(lines)


def solve(puzzle: Puzzle, *, max_worlds: int = MAX_WORLDS) -> Solution | Unknown:
    """Enumerate the candidate worlds and eliminate the ones a constraint rules out.

    Refuses rather than hangs: the number of worlds is the product of the domain sizes, and a
    product over ``max_worlds`` comes back as ``Unknown("too_many_worlds")`` before anything
    is enumerated. An abstention rather than an exception because that is what an operation
    unable to answer returns here — a cascade can then try something else, and a caller that
    ignores it gets an ``Unknown`` it cannot accidentally read as an answer.
    """
    if puzzle.size > max_worlds:
        sizes = " x ".join(f"{len(v)}" for v in puzzle.variables)
        return Unknown(
            "too_many_worlds",
            f"{sizes} = {puzzle.size:,} candidate worlds, over the cap of {max_worlds:,}; "
            f"enumeration costs O(worlds x constraints), so this would not finish. "
            f"Cut the questions down or raise max_worlds deliberately.",
        )
    surviving: list[World] = []
    eliminated: list[Elimination] = []
    undecided: list[Elimination] = []
    for world in puzzle.worlds():
        ruled_out: Elimination | None = None
        blocked: Elimination | None = None  # the first constraint that could not be decided
        for constraint in puzzle.constraints:
            verdict = constraint.judge(world)
            if verdict.status == "fails":
                ruled_out = Elimination(world, constraint.name, verdict.reasons)
                break
            if verdict.status == "unknown" and blocked is None:
                blocked = Elimination(world, constraint.name, verdict.reasons)
        if ruled_out is not None:
            eliminated.append(ruled_out)
        elif blocked is not None:
            undecided.append(blocked)
        else:
            surviving.append(world)
    return Solution(puzzle, tuple(surviving), tuple(eliminated), tuple(undecided))


def entails(proposition: Proposition, solution: Solution) -> Verdict:
    """Does ``proposition`` hold in every world that survived? The trace comes with the verdict.

    This is the shape the rest of the library already has a word for — a three-state
    :class:`~tensorcode.outcomes.Verdict` — and it is what makes the solver answerable through
    ``ops.check``: "is the prize in box b?" is a claim to evaluate against evidence, where the
    evidence is the constraints. True in some survivors and false in others is ``unknown``,
    never a majority vote.
    """
    trace = tuple(solution.explain().splitlines())
    if solution.undecided:
        return Verdict("unknown", ("some candidates could not be judged",) + trace)
    if not solution.surviving:
        return Verdict("unknown", ("no world satisfies the constraints",) + trace)
    truths = {solution.puzzle.truth(world, proposition) for world in solution.surviving}
    if truths == {True}:
        return Verdict("holds", (f"{_plain(proposition)} is true in all {len(solution.surviving)} surviving world(s)",) + trace)
    if truths == {False}:
        return Verdict("fails", (f"{_plain(proposition)} is false in all {len(solution.surviving)} surviving world(s)",) + trace)
    return Verdict("unknown", (f"{_plain(proposition)} is not settled by the surviving worlds",) + trace)


def _accepts_a_puzzle(request: Request) -> bool:
    return isinstance(request.subject, Proposition) and any(
        isinstance(e, Puzzle) for e in request.params.get("evidence", ())
    )


@implementation(
    "check",
    name="check:world-elimination",
    version="1",
    accepts=_accepts_a_puzzle,
    traits=Traits(locality="in_process", deterministic=True),
    profile=Profile(),  # unmeasured: no assay task exercises it yet, and an unmeasured
    # implementation must not look good, or a policy ordering by quality orders by zero
)
def check_by_elimination(request: Request) -> Any:
    """``check(proposition, evidence=[puzzle])``: is it true in every world the puzzle allows?

    Registering the solver as a ``check`` implementation rather than a new operation is the
    honest seam. The agent already routes its turn through ``parse`` / ``choose`` / ``rank``
    / ``verify``, and nothing about a puzzle needs a new verb: the question "which box holds
    the prize" is a ``check`` of a candidate claim against evidence, so it inherits the
    trace, the cascade, and the policy for free, and a host that would rather ask a model
    can register another implementation of the same operation.
    """
    puzzles = [e for e in request.params.get("evidence", ()) if isinstance(e, Puzzle)]
    if len(puzzles) != 1:
        return Unknown("ambiguous_evidence", f"{len(puzzles)} puzzles offered as evidence; one is needed")
    solution = solve(puzzles[0], **({"max_worlds": request.params["max_worlds"]} if "max_worlds" in request.params else {}))
    if isinstance(solution, Unknown):
        return solution
    return entails(request.subject, solution)
