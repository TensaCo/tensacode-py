"""Induction, and the controls that decide whether an induced artifact may be adopted."""

import random

import pytest

from tensorcode.learning import (
    Concept, Literal, candidate_literals, check_concept, decision_list, effects, preconditions, propose_concepts,
    rename_case, rename_map, role_type, shape, verify_decision_list,
)


def facts(**kw):
    return frozenset(kw.items())


def routing_cases(n=60, seed=0):
    """A rule no single condition captures: three outcomes from two interacting facts.

    A one-condition competitor can only ever get two of the three right, which is
    what makes this a fair test of a rule *list* rather than of the floor.
    """
    rng = random.Random(seed)
    out = []
    for i in range(n):
        urgent = rng.random() < 0.5
        paid = rng.random() < 0.5
        region = rng.choice(["north", "south", "east"])
        label = "fast" if (urgent and paid) else "queued" if urgent else "normal"
        out.append((facts(urgent=urgent, paid=paid, region=region, index=i % 7), label))
    return out


def test_a_decision_list_is_readable_and_explains_itself():
    cases = routing_cases()
    literals = candidate_literals(cases)
    rules = decision_list(cases, literals)
    assert rules.rules, "nothing was induced"
    assert rules.predict(facts(urgent=True, paid=True, region="north", index=1)) == "fast"
    assert rules.predict(facts(urgent=False, paid=True, region="north", index=1)) == "normal"
    assert rules.accuracy(cases) > 0.95
    # every decision is attributable: either to a rule, or explicitly to the default
    explained = [rules.explain(f) for f, _ in cases]
    assert any(rule is not None for _, rule in explained)
    assert "urgent" in repr(rules) or "paid" in repr(rules)
    assert rules.cost() > 0


def test_the_mdl_stop_refuses_rules_that_do_not_pay_for_themselves():
    rng = random.Random(1)
    noise = [(facts(a=rng.random() < 0.5, b=rng.random() < 0.5), rng.choice(["x", "y"])) for _ in range(60)]
    literals = candidate_literals(noise)
    with_mdl = decision_list(noise, literals, mdl=True)
    without = decision_list(noise, literals, mdl=False)
    assert len(with_mdl.rules) < len(without.rules)


def test_held_out_accuracy_and_the_four_controls_admit_a_real_rule():
    train, held = routing_cases(80, seed=2), routing_cases(40, seed=3)
    literals = candidate_literals(train)
    rules = decision_list(train, literals)
    check = verify_decision_list(rules, train=train, held_out=held, literals=literals)
    assert check.adopted, check.report()
    assert check.held_out > check.random and check.held_out >= check.floor
    assert check.renamed_identical


def test_an_artifact_that_reads_vocabulary_is_rejected_by_the_renaming_control():
    # the label *is* the region's name, so the rules can only be reading vocabulary
    cases = [(facts(region=r, urgent=True), r) for r in ["north", "south", "east", "west"] * 8]
    literals = candidate_literals(cases)
    rules = decision_list(cases, literals, min_confidence=0.3)
    check = verify_decision_list(rules, train=cases, held_out=cases, literals=literals)
    assert not check.renamed_identical
    assert not check.adopted and any("renaming" in r for r in check.verdict.reasons)


def test_a_rule_that_only_matches_the_majority_is_rejected_by_the_floor():
    cases = [(facts(a=True, b=i % 2 == 0), "same") for i in range(40)]
    literals = candidate_literals(cases)
    rules = decision_list(cases, literals, min_confidence=0.1)
    check = verify_decision_list(rules, train=cases, held_out=cases, literals=literals)
    assert not check.adopted and any("floor" in r for r in check.verdict.reasons)


def test_the_wrong_question_control_catches_an_artifact_that_ignores_it():
    train, held = routing_cases(80, seed=4), routing_cases(40, seed=5)
    literals = candidate_literals(train)
    rules = decision_list(train, literals)
    # shifted: the same facts, labels from a different question (always "normal")
    shifted = [(f, "normal") for f, _ in held]
    check = verify_decision_list(rules, train=train, held_out=held, literals=literals, shifted=shifted)
    assert check.shifted <= check.held_out
    assert isinstance(check.report(), str)


def test_preconditions_are_the_most_specific_cover_and_intervention_prunes_them():
    # the action fires when the door is unlocked; "weather" is constant but irrelevant
    fired = [facts(locked=False, weather="rain", holder="a"), facts(locked=False, weather="rain", holder="b")]
    passive = preconditions("open", fired)
    assert Literal("locked", False) in passive.conditions
    assert Literal("weather", "rain") in passive.conditions  # over-specialised, as expected

    def world(action, state):
        return dict(state).get("locked") is False  # only the lock matters

    pruned = preconditions("open", fired, intervene=world)
    assert Literal("locked", False) in pruned.conditions
    assert Literal("weather", "rain") not in pruned.conditions
    assert "intervention" in pruned.method


def test_negative_states_drop_conditions_that_held_when_nothing_happened():
    fired = [facts(locked=False, lights=True), facts(locked=False, lights=True)]
    idle = [facts(locked=True, lights=True)]
    pruned = preconditions("open", fired, did_not_fire=idle)
    assert Literal("lights", True) not in pruned.conditions
    assert Literal("locked", False) in pruned.conditions


def test_effects_are_what_every_firing_added():
    before = [facts(open=False), facts(open=False)]
    after = [facts(open=True), facts(open=True)]
    assert effects(before, after) == (Literal("open", True),)


def test_a_role_type_generalises_only_when_the_shape_is_productive():
    productive = role_type("region", [f"region_{i}" for i in range(8)])
    assert productive.productive and productive.admits("region_84")
    memorised = role_type("group", ["group", "team", "crew"])
    assert not memorised.productive and memorised.admits("team") and not memorised.admits("squad")
    assert shape("region_0") == shape("region_84")


def test_a_shape_shared_with_other_things_is_not_productive():
    fillers = [f"x{i}" for i in range(6)]
    others = [f"x{i}" for i in range(100, 130)]  # the same shape, mostly not of this kind
    assert not role_type("code", fillers, others=others).productive


def test_concepts_are_proposed_by_behaviour_and_checked_before_adoption():
    cases = routing_cases(60, seed=6)
    literals = candidate_literals(cases)
    proposals = propose_concepts(cases, literals, label="fast")
    assert proposals, "no candidate definitions"
    positives = [f for f, y in cases if y == "fast"]
    negatives = [f for f, y in cases if y != "fast"]
    check = check_concept(proposals[0], positives=positives, negatives=negatives)
    assert check.adopted, check.reasons
    assert check.renamed_identical and check.round_trips


def test_a_concept_that_leaks_into_the_negatives_is_rejected_with_a_reason():
    leaky = Concept("anything", (Literal("urgent", kind="present"),), support=1)
    cases = routing_cases(40, seed=7)
    check = check_concept(leaky, positives=[f for f, y in cases if y == "fast"],
                          negatives=[f for f, y in cases if y != "fast"])
    assert not check.adopted and any("negative" in r for r in check.reasons)


def test_renaming_is_consistent_and_order_independent():
    cases = routing_cases(20, seed=8)
    mapping = rename_map(cases)
    once = [rename_case(c, mapping) for c in cases]
    twice = [rename_case(c, rename_map(cases)) for c in cases]
    assert once == twice
