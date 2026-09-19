"""Stated-constraint puzzles: the answer, and the eliminations that earn it.

The owner asked the agent the three-boxes puzzle — exactly one prize, each box carrying a
statement, exactly one of the statements true — and got "I don't know" and "I can not explain
my reasoning". These tests pin down the machinery that fixes it, and they are written against
propositions built by hand: reading English into propositions is the readers' job, and a
solver that only works when a particular sentence is phrased a particular way would be the
puzzle's answer written as code (docs/revival/32).

Each test names the failure it guards, because most of them are ways of *seeming* to reason:
answering without the eliminations, eliminating a candidate on a question nobody asked,
picking one of two survivors, or hanging on a space too large to enumerate.
"""

from __future__ import annotations

import time

import pytest

import tensorcode as tc
from tensorcode.records import Proposition
from tensorcode.reasoning import (
    Distinct,
    MustHold,
    Puzzle,
    TruthCount,
    Variable,
    entails,
    exactly_one,
    one_of,
    solve,
)
from tensorcode.reasoning.constraints import check_by_elimination


def holds(box: str, *, what: str = "prize", polarity: bool = True) -> Proposition:
    """"the ``what`` is (not) in box ``box``" — one predication, no reified event."""
    return Proposition(
        "contains",
        {"container": tc.Ref(f"box:{box}"), "content": tc.Ref(f"thing:{what}")},
        polarity=polarity,
    )


def three_boxes() -> Puzzle:
    """The owner's probe, encoded.

    ``a: 'the prize is not in this box'``, ``b: 'the prize is in box a'``,
    ``c: 'the prize is not in box b'``, exactly one of the three true. "Exactly one box
    contains a prize" is the *question*: one variable whose three alternatives exclude each
    other. The statements are not asserted anywhere — they are what the count is about.
    """
    prize = one_of("prize", holds("a"), holds("b"), holds("c"))
    return Puzzle(
        (prize,),
        (
            exactly_one(
                holds("a", polarity=False),
                holds("a"),
                holds("b", polarity=False),
                label="exactly one of the three statements is true",
            ),
        ),
    )


# ------------------------------------------------------- the puzzle it could not do


def test_three_boxes_has_one_answer_and_it_is_box_b():
    """The probe itself. Before this, the reply was 'I don't know'."""
    solution = solve(three_boxes())
    assert not isinstance(solution, tc.Unknown)
    assert solution.considered == 3
    assert solution.unique
    answer = solution.answer()
    assert not isinstance(answer, tc.Unknown)
    assert answer["prize"].roles["container"] == tc.Ref("box:b")


def test_the_answer_comes_with_which_candidate_each_constraint_ruled_out():
    """'I can not explain my reasoning' was the other half of the failure.

    A solver that returns only the survivor is not acceptable here: the owner asked it to
    show its work, and an answer nobody can check is indistinguishable from a lucky guess.
    """
    solution = solve(three_boxes())
    assert [e.world["prize"].roles["container"] for e in solution.eliminated] == [
        tc.Ref("box:a"),
        tc.Ref("box:c"),
    ]
    assert {e.constraint for e in solution.eliminated} == {"exactly one of the three statements is true"}
    # each elimination says what the count actually came to, and which statements made it
    ruled_out_a, ruled_out_c = solution.eliminated
    assert ruled_out_a.reasons[0].startswith("2 true")
    assert "contains(container=box:a, content=thing:prize)" in ruled_out_a.reasons[1]  # b's statement was true
    assert "not contains(container=box:b, content=thing:prize)" in ruled_out_c.reasons[1]
    trace = solution.explain()
    assert "3 candidate worlds from 1 open question: 1 surviving, 2 ruled out" in trace
    assert trace.count("\n  ruled out") == 2 and trace.count("\n  survives") == 1


def test_a_denial_is_decided_by_exclusivity_not_by_absence():
    """"The prize is not in this box" is true because the *question* was settled elsewhere.

    If the solver treated an unsupported proposition as false, ``not contains(box:a)`` would
    come out true in the world that puts the prize in a — the puzzle would have no answer at
    all, and the bug would look like the puzzle being unsolvable.
    """
    puzzle = three_boxes()
    in_a, in_b, _ = puzzle.variables[0].domain
    world_a = next(w for w in puzzle.worlds() if w["prize"] == in_a)
    world_b = next(w for w in puzzle.worlds() if w["prize"] == in_b)
    assert world_a.truth(holds("a", polarity=False)) is False
    assert world_b.truth(holds("a", polarity=False)) is True
    assert world_b.truth(holds("c")) is False  # a passed-over alternative is false, not unknown


def test_hypotheses_go_into_the_store_without_being_believed():
    """A candidate world is storable beside its rivals; that is what ``hypothesised`` is for.

    The point of building on the claim schema rather than beside it: the three locations can
    sit in one store, and retrieval for what the world *is* must not return any of them.
    """
    store = tc.Store(tc.TypeRegistry())
    evidence = tc.Evidence(tc.Ref("obs:puzzle"), tc.records.datetime.now(tc.records.timezone.utc))
    for hypothesis in three_boxes().variables[0].domain:
        assert hypothesis.modality == "hypothesised"
        store.assert_(hypothesis, evidence)
    assert store.find(Proposition("contains", {"content": tc.Ref("thing:prize")})) == []
    held = store.find(Proposition("contains", {"content": tc.Ref("thing:prize")}, modality="hypothesised"))
    assert len(held) == 3


# ------------------------------------------------------------- other puzzle shapes


def test_two_statements_leave_two_worlds_and_it_refuses_to_pick():
    """Drop one statement and the puzzle stops determining anything.

    The dangerous failure is not getting this wrong, it is getting it *confidently*: with two
    survivors, the only honest answer is both of them and the count.
    """
    prize = one_of("prize", holds("a"), holds("b"), holds("c"))
    puzzle = Puzzle(
        (prize,),
        (exactly_one(holds("a", polarity=False), holds("b", polarity=False), label="exactly one of the two is true"),),
    )
    solution = solve(puzzle)
    assert not solution.unique
    assert [w["prize"].roles["container"] for w in solution.surviving] == [tc.Ref("box:a"), tc.Ref("box:b")]
    assert [e.world["prize"].roles["container"] for e in solution.eliminated] == [tc.Ref("box:c")]
    answer = solution.answer()
    assert isinstance(answer, tc.Unknown)
    assert answer.reason == "ambiguous"
    assert "box:a" in answer.detail and "box:b" in answer.detail


def test_four_boxes_two_questions_and_a_relation_between_the_answers():
    """A bigger shape: four alternatives, a second question, and a constraint relating them.

    Four statements about the prize (``in b``, ``not in b``, ``not in a``, ``in d``) with
    exactly one true leave only ``box a``; the key is then placed by "not in the box with the
    prize", which is a relation between two answers and cannot be a fact about either.
    """
    prize = one_of("prize", *(holds(box) for box in "abcd"))
    key = one_of("key", holds("a", what="key"), holds("b", what="key"))
    puzzle = Puzzle(
        (prize, key),
        (
            exactly_one(
                holds("b"),
                holds("b", polarity=False),
                holds("a", polarity=False),
                holds("d"),
                label="exactly one of the four statements is true",
            ),
            Distinct(("prize", "key"), "container", label="the key is not in the box with the prize"),
        ),
    )
    solution = solve(puzzle)
    assert solution.considered == 8  # 4 x 2, the product of the domains
    assert solution.unique
    answer = solution.answer()
    assert answer["prize"].roles["container"] == tc.Ref("box:a")
    assert answer["key"].roles["container"] == tc.Ref("box:b")
    by_constraint = [e.constraint for e in solution.eliminated]
    assert by_constraint.count("exactly one of the four statements is true") == 6
    assert by_constraint.count("the key is not in the box with the prize") == 1
    collision = next(e for e in solution.eliminated if e.constraint.startswith("the key"))
    assert collision.reasons == ("key and prize share container=box:a",)


def test_a_stated_fact_eliminates_on_its_own():
    """``MustHold`` is the plain case, and polarity carries the denial."""
    prize = one_of("prize", holds("a"), holds("b"), holds("c"))
    solution = solve(Puzzle((prize,), (MustHold(holds("a", polarity=False)), MustHold(holds("c", polarity=False)))))
    assert solution.unique
    assert solution.answer()["prize"].roles["container"] == tc.Ref("box:b")
    assert [e.constraint for e in solution.eliminated] == [
        "must hold: not contains(container=box:a, content=thing:prize)",
        "must hold: not contains(container=box:c, content=thing:prize)",
    ]


# ------------------------------------------------- what it must refuse to conclude


def test_an_undetermined_statement_eliminates_nothing():
    """A statement about a box no question ranges over is unknown, and unknown is not false.

    Counting it as false would hand back a unique answer resting on a question nobody asked.
    The worlds stay in ``undecided``, and ``answer()`` says so instead of naming a box.
    """
    prize = one_of("prize", holds("a"), holds("b"), holds("c"))
    puzzle = Puzzle(
        (prize,),
        (exactly_one(holds("a", polarity=False), holds("a"), holds("e"), label="exactly one of the three is true"),),
    )
    solution = solve(puzzle)
    assert solution.eliminated == ()
    assert len(solution.undecided) == 3
    assert not solution.unique
    answer = solution.answer()
    assert isinstance(answer, tc.Unknown)
    assert answer.reason == "undetermined"
    assert "undetermined: contains(container=box:e, content=thing:prize)" in solution.undecided[0].reasons[-1]


def test_a_count_can_still_decide_when_a_member_is_undetermined():
    """Partial information narrows the count to an interval, which is often enough.

    Two members already true breaks ``at most one`` whatever the third turns out to be, so
    the solver draws the conclusion rather than giving up whenever anything is unknown.
    """
    prize = one_of("prize", holds("a"), holds("b"), holds("c"))
    puzzle = Puzzle(
        (prize,),
        (
            TruthCount(
                (holds("a", polarity=False), holds("b", polarity=False), holds("e")),
                at_most=1,
                label="at most one of the three is true",
            ),
        ),
    )
    solution = solve(puzzle)
    # prize in c makes both denials true: 2 already, whatever box e holds
    assert [e.world["prize"].roles["container"] for e in solution.eliminated] == [tc.Ref("box:c")]
    assert solution.eliminated[0].reasons[0].startswith("between 2 and 3 true")
    assert len(solution.undecided) == 2  # a and b are 1-or-2, which this bound cannot settle


def test_it_refuses_a_space_it_cannot_enumerate_instead_of_hanging():
    """The bound is the product of the domain sizes, checked before anything is enumerated.

    Refusing in O(number of questions) is the point: an over-encoded puzzle should say so
    immediately, not appear to be thinking for an hour.
    """
    variables = tuple(
        Variable(f"q{i}", tuple(holds(f"b{j}", what=f"prize{i}") for j in range(10))) for i in range(12)
    )
    puzzle = Puzzle(variables, (MustHold(holds("b0", what="prize0")),))
    assert puzzle.size == 10**12
    started = time.perf_counter()
    refusal = solve(puzzle)
    assert (time.perf_counter() - started) < 1.0
    assert isinstance(refusal, tc.Unknown)
    assert refusal.reason == "too_many_worlds"
    assert "1,000,000,000,000 candidate worlds" in refusal.detail
    assert "O(worlds x constraints)" in refusal.detail
    # the cap is a decision, not a limit: a caller who means it can raise it
    assert not isinstance(solve(Puzzle(variables[:2], ()), max_worlds=100), tc.Unknown)


def test_two_questions_over_the_same_proposition_are_rejected():
    """An encoding where two questions could answer the same thing differently is a bug.

    Whichever was consulted first would silently overrule the other, so the puzzle refuses to
    be built rather than producing an answer that depends on variable order.
    """
    with pytest.raises(ValueError, match="both range over"):
        Puzzle((one_of("x", holds("a"), holds("b")), one_of("y", holds("b"), holds("c"))))


def test_a_desire_is_not_a_candidate_for_being_true():
    """Modality is not decoration: folding ``desired`` into truth would prove wishes."""
    with pytest.raises(ValueError, match="not a candidate for being true"):
        one_of("prize", Proposition("contains", {"container": tc.Ref("box:a")}, modality="desired"))


# ------------------------------------------------------------------ the ops seam


def test_a_candidate_claim_is_answerable_through_ops_check():
    """The solver's seam into the agent: ``check(claim, evidence=[puzzle])``.

    A puzzle needs no new verb. "Is the prize in box b?" is a claim evaluated against
    evidence, so it goes through the ``check`` family and inherits the trace, the cascade and
    the policy — and the verdict carries the elimination trace as its reasons.
    """
    puzzle = three_boxes()
    with tc.use(tc.Runtime([check_by_elimination])) as runtime:
        yes = tc.check(holds("b"), evidence=[puzzle])
        no = tc.check(holds("a"), evidence=[puzzle])
    assert yes.status == "holds"
    assert no.status == "fails"
    assert any("ruled out" in reason for reason in yes.reasons)
    assert runtime.trace.of("check")[0].answered_by == "check:world-elimination@1"


def test_check_abstains_rather_than_answering_from_an_unsolvable_space():
    """An abstention travels as ``Unknown`` so a cascade can try something else."""
    puzzle = Puzzle(
        tuple(Variable(f"q{i}", tuple(holds(f"b{j}", what=f"p{i}") for j in range(10))) for i in range(9)),
        (MustHold(holds("b0", what="p0")),),
    )
    with tc.use(tc.Runtime([check_by_elimination])):
        verdict = tc.check(holds("b0", what="p0"), evidence=[puzzle])
    assert verdict.status == "unknown"
    assert "too_many_worlds" in verdict.reasons[0]


def test_entailment_over_two_survivors_is_unknown_not_a_majority():
    """With two worlds left, a claim true in one of them is not true."""
    prize = one_of("prize", holds("a"), holds("b"), holds("c"))
    puzzle = Puzzle(
        (prize,),
        (exactly_one(holds("a", polarity=False), holds("b", polarity=False)),),
    )
    solution = solve(puzzle)
    assert entails(holds("a"), solution).status == "unknown"
    assert entails(holds("c"), solution).status == "fails"  # false in both survivors
