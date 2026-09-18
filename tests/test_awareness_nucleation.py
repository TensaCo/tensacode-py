"""Awareness: seeding, spreading along real links, bounding, explaining, and fading."""

from datetime import datetime, timezone

import tensorcode as tc
from tensorcode.awareness import Awareness, AwarenessPolicy, nucleate
from tensorcode.cognition import Rule, Thought, think

NOW = datetime(2026, 9, 17, 12, 0, tzinfo=timezone.utc)


def seen(mind, subject, predicate, obj, source="obs:frame-1", method="dom-scene-graph"):
    claim = tc.Claim(tc.Ref(subject), predicate, obj)
    return mind.tell(claim, tc.Evidence(tc.Ref(source), NOW, method=method))


def test_awareness_spreads_from_a_seed_to_the_same_subject_and_stops_at_the_floor():
    mind = tc.Store()
    dock = seen(mind, "ui:dock/button/Terminal", "label", "Terminal")
    seen(mind, "ui:dock/button/Terminal", "is_a", "button")
    far = seen(mind, "path:/home/agent/Documents/unrelated.txt", "is_a", "regular file", source="obs:other")

    a = Awareness(mind, AwarenessPolicy(budget=10, decay=0.6, floor=0.2, max_hops=2))
    a.seed(dock.id)
    a.spread()
    ids = a.aware_ids()

    assert dock.id in ids, "the seed is aware"
    assert any(r.claim.predicate == "is_a" and r.claim.subject == dock.claim.subject for r in a.aware()), "same-subject claim was pulled in"
    assert far.id not in ids, "an unlinked claim in another observation stays out"
    assert a.salience(dock.id) > a.salience(next(r.id for r in a.aware() if r.id != dock.id))


def test_activation_is_explainable_as_a_path_from_its_seed():
    mind = tc.Store()
    root = seen(mind, "message:1", "text", "what is on my desktop")
    linked = seen(mind, "message:1", "from", "user")

    a = nucleate(mind, [root.id], AwarenessPolicy(budget=8, floor=0.1))
    lines = a.why(linked.id)

    assert "message:1 text" in lines[0]
    assert any("--subject-->" in line for line in lines), lines
    assert a.why("claim:absent") == ["claim:absent is not aware"]


def test_provenance_links_make_a_derivation_aware_with_its_premise():
    """A conclusion about one thing reaches the observation it rests on, about another."""
    mind = tc.Store()
    premise = seen(mind, "ui:field#1", "label", "Staff number")
    derived = mind.tell(
        tc.Claim(tc.Ref("form:access-request"), "needs", "employee_id"),
        tc.Evidence(tc.Ref("rule:label_means"), NOW, method="derive@1", derived_from=(premise.id,)),
    )

    a = nucleate(mind, [derived.id], AwarenessPolicy(budget=8, floor=0.1))

    assert premise.id in a.aware_ids(), "provenance is a link awareness travels"
    assert any("--premise-->" in line for line in a.why(premise.id)), a.why(premise.id)


def test_the_budget_bounds_what_is_aware_however_big_memory_is():
    mind = tc.Store()
    hub = tc.Ref("window:Files")
    first = seen(mind, "window:Files", "shows", "row 0")
    for i in range(1, 200):
        mind.tell(tc.Claim(hub, "shows", f"row {i}"), tc.Evidence(tc.Ref("obs:frame-1"), NOW, method="dom-scene-graph"))

    a = nucleate(mind, [first.id], AwarenessPolicy(budget=12, floor=0.01, max_hops=3))

    assert len(a.aware()) == 12, "the aware set is bounded by the budget, not by memory"
    assert len(mind.claims()) == 200


def test_thinking_over_awareness_costs_the_aware_set_not_the_store():
    """The quantitative case: a rule whose pattern scans everything is bounded by awareness."""
    mind = tc.Store()
    focus = seen(mind, "path:/home/agent/Desktop/notes.txt", "is_a", "regular file")
    for i in range(400):
        mind.tell(tc.Claim(tc.Ref(f"path:/old/file{i}.txt"), "is_a", "regular file"),
                  tc.Evidence(tc.Ref("obs:sweep"), NOW, method="dom-scene-graph"))

    fired: list[str] = []
    rule = Rule("anything_is_a", ((tc.Var("x"), "is_a", tc.Var("k")),),
                lambda b, m: fired.append(b["x"].id) or [])

    a = nucleate(mind, [focus.id], AwarenessPolicy(budget=8, floor=0.2, max_hops=1))
    view = a.view()
    think(view, [rule], since=Thought(added=tuple(view.claims())))
    bounded = len(fired)

    fired.clear()
    think(mind, [rule], since=Thought(added=tuple(mind.claims())))
    unbounded = len(fired)

    assert bounded <= 8 and unbounded == 401
    assert unbounded > bounded * 10, f"awareness bounded the rule to {bounded} firings against {unbounded}"


def test_fading_drops_what_is_no_longer_active_and_keeps_what_was_strong():
    mind = tc.Store()
    strong = seen(mind, "ui:button#1", "label", "Send")
    weak = seen(mind, "ui:button#1", "is_a", "button")

    a = Awareness(mind, AwarenessPolicy(budget=10, decay=0.5, floor=0.3, fade=0.5))
    a.seed(strong.id)
    a.spread()
    assert {strong.id, weak.id} <= a.aware_ids() or weak.id not in a.aware_ids()

    dropped = a.fade()
    assert strong.id in a.aware_ids() or strong.id in dropped
    a.fade()
    a.fade()
    assert not a.aware_ids(), "activation decays away when nothing renews it"


def test_a_retracted_claim_stops_being_aware():
    mind = tc.Store()
    rec = seen(mind, "ui:toast#1", "announces", "Saved")
    a = nucleate(mind, [rec.id], AwarenessPolicy(budget=4))
    assert rec.id in a.aware_ids()

    mind.apply(tc.Patch((tc.Retract(rec.id, "no longer perceived"),), mind.revision))

    assert rec.id not in a.aware_ids()
    assert a.salience(rec.id) == 0.0


def test_awareness_is_deterministic():
    mind = tc.Store()
    seeds = [seen(mind, f"ui:control#{i}", "in", "Files").id for i in range(6)]
    for i in range(6):
        seen(mind, f"ui:control#{i}", "label", f"Item {i}")

    first = nucleate(mind, seeds[:1], AwarenessPolicy(budget=6, floor=0.05))
    second = nucleate(mind, seeds[:1], AwarenessPolicy(budget=6, floor=0.05))

    assert [r.id for r in first.aware()] == [r.id for r in second.aware()]
    assert [first.salience(r.id) for r in first.aware()] == [second.salience(r.id) for r in second.aware()]
