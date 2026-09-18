import tensacode as tc
from tensacode.cognition import Fragment, Rule, Thought, explain, integrate, think

SCREEN = tc.Ref("scope:screen")


def frame(n, *claims):
    return Fragment(tc.Ref(f"obs:frame-{n}"), tuple((c, f"box{i}") for i, c in enumerate(claims)), snapshot_of=SCREEN)


def seen(subject, predicate, obj):
    return tc.Claim(tc.Ref(subject), predicate, obj, scope=SCREEN)


LABEL_MEANS = Rule(
    "label_means",
    ((tc.Var("f"), "label", tc.Var("l")),),
    lambda b, mind: [tc.Claim(b["f"], "means", "employee_id")] if "staff" in b["l"].lower() else [],
)


def test_snapshot_retracts_what_is_no_longer_perceived():
    mind = tc.Store()
    t1 = integrate(mind, frame(1, seen("control:a", "label", "Staff number"), seen("control:b", "label", "Name")))
    assert len(t1.added) == 2
    t2 = integrate(mind, frame(2, seen("control:a", "label", "Staff number")))
    assert [r.claim.subject.id for r in t2.retracted] == ["control:b"] and not t2.added
    assert integrate(mind, frame(3, seen("control:a", "label", "Staff number"))).empty
    assert len(mind.claims()) == 1  # retracted perception is forgotten, not hoarded


def test_think_derives_with_provenance_and_is_semi_naive():
    mind = tc.Store()
    t = integrate(mind, frame(1, seen("control:a", "label", "Staff number")))
    fired = []
    rule = Rule(LABEL_MEANS.name, LABEL_MEANS.when, lambda b, m: fired.append(1) or LABEL_MEANS.then(b, m))
    d = think(mind, [rule], since=t)
    assert [r.claim.predicate for r in d.added] == ["means"] and fired == [1]
    assert think(mind, [rule], since=Thought()).empty and fired == [1]  # nothing new: nothing fires
    (derived,) = mind.claims(predicate="means")
    lines = explain(mind, derived.id)
    assert lines[0] == "control:a means 'employee_id'" and "rule:label_means" in lines[1] and "observed in obs:frame-1 at box0" in "\n".join(lines)


def test_derivations_fall_away_with_their_premises():
    mind = tc.Store()
    think(mind, [LABEL_MEANS], since=integrate(mind, frame(1, seen("control:a", "label", "Staff number"))))
    assert mind.claims(predicate="means")
    t = integrate(mind, frame(2))
    assert {r.claim.predicate for r in t.retracted} == {"label", "means"}
    assert mind.claims(predicate="means") == []


def test_derivation_survives_if_independently_supported():
    mind = tc.Store()
    think(mind, [LABEL_MEANS], since=integrate(mind, frame(1, seen("control:a", "label", "Staff number"))))
    (derived,) = mind.claims(predicate="means")
    mind.tell(derived.claim, tc.Evidence(tc.Ref("obs:human"), derived.evidence[0].observed_at))  # also asserted directly
    integrate(mind, frame(2))
    assert mind.claims(predicate="means")


def test_chained_rules_reach_a_fixpoint():
    mind = tc.Store()
    needs_value = Rule("needs_value", ((tc.Var("f"), "means", "employee_id"),), lambda b, m: [tc.Claim(b["f"], "should_contain", "E12345")])
    t = integrate(mind, frame(1, seen("control:a", "label", "Staff number")))
    d = think(mind, [LABEL_MEANS, needs_value], since=t)
    assert [r.claim.predicate for r in d.added] == ["means", "should_contain"]
    assert "should_contain" in explain(mind, d.added[1].id)[0] and "means" in "\n".join(explain(mind, d.added[1].id))


def test_match_variables_must_agree_across_patterns():
    mind = tc.Store()
    for s, p, o in (("a", "likes", "b"), ("b", "likes", "a"), ("b", "likes", "c")):
        mind.tell(tc.Claim(tc.Ref(f"x:{s}"), p, tc.Ref(f"x:{o}")), tc.Evidence(tc.Ref("obs:t"), derived_now()))
    mutual = mind.match((tc.Var("p"), "likes", tc.Var("q")), (tc.Var("q"), "likes", tc.Var("p")))
    assert sorted((m["p"].id, m["q"].id) for m in mutual) == [("x:a", "x:b"), ("x:b", "x:a")]


def derived_now():
    from datetime import datetime, timezone

    return datetime.now(timezone.utc)


def test_rules_can_react_to_disappearance():
    mind = tc.Store()
    mind.tell(tc.Claim(tc.Ref("agent:me"), "watching", True), tc.Evidence(tc.Ref("obs:x"), derived_now()))
    integrate(mind, frame(1, seen("ui:spinner", "shows", "…")))
    noticed = Rule(
        "spinner_gone",
        ((tc.Ref("agent:me"), "watching", True),),
        lambda b, m: [] if m.claims(tc.Ref("ui:spinner")) else [tc.Claim(tc.Ref("job:1"), "finished", True)],
        reacts_to="any_change",
    )
    gone = integrate(mind, frame(2))  # only a retraction
    assert not gone.added and gone.retracted
    assert [r.claim.predicate for r in think(mind, [noticed], since=gone).added] == ["finished"]
    assert think(mind, [LABEL_MEANS], since=gone).empty  # ordinary rules still need new premises


# ------------------------------------------------------------ corroboration


from tensacode.cognition import Corroboration  # noqa: E402


def reads(frame_no, text, conf=None, subject="text:line#1"):
    c = seen(subject, "reads", text)
    return Fragment(tc.Ref(f"obs:frame-{frame_no}"), ((c, "box"),), snapshot_of=SCREEN, confidence={c.id: tc.Score(conf, "uncalibrated")} if conf is not None else {})


def test_a_reading_becomes_established_only_when_seen_in_k_frames():
    mind, policy = tc.Store(), Corroboration(k=2)
    t1 = integrate(mind, reads(1, "task-71008.txt"), corroboration=policy)
    (rec,) = t1.added
    assert policy.tentative(mind, rec.id) and policy.view(mind).claims(predicate="reads") == []
    t2 = integrate(mind, reads(2, "task-71008.txt"), corroboration=policy)
    assert not t2.added and [r.id for r in t2.established] == [rec.id]
    assert policy.view(mind).claims(predicate="reads")[0].claim.object == "task-71008.txt"


def test_a_contradicted_glance_is_retracted_and_its_replacement_starts_over():
    mind, policy = tc.Store(), Corroboration(k=2)
    integrate(mind, reads(1, "task-71088.txt"), corroboration=policy)
    t2 = integrate(mind, reads(2, "task-71008.txt"), corroboration=policy)
    assert [r.claim.object for r in t2.retracted] == ["task-71088.txt"] and not t2.established
    assert [r.claim.object for r in mind.claims(predicate="reads")] == ["task-71008.txt"]
    assert policy.view(mind).claims(predicate="reads") == []  # one frame of the new reading is not enough
    assert integrate(mind, reads(3, "task-71008.txt"), corroboration=policy).established


def test_non_snapshot_functional_contradiction_retracts_the_tentative_claim():
    mind, policy = tc.Store(), Corroboration(k=2)
    mind.declare("status", functional=True)
    first = tc.Claim(tc.Ref("job:1"), "status", "running")
    integrate(mind, Fragment(tc.Ref("obs:a"), ((first, None),)), corroboration=policy)
    later = tc.Claim(tc.Ref("job:1"), "status", "done")
    t = integrate(mind, Fragment(tc.Ref("obs:b"), ((later, None),)), corroboration=policy)
    assert [r.claim.object for r in t.retracted] == ["running"]
    assert [r.claim.object for r in mind.claims(predicate="status")] == ["done"]


def test_derivations_inherit_tentativeness_and_strict_rules_wait():
    mind, policy = tc.Store(), Corroboration(k=2, predicates=frozenset({"reads"}))
    note_file = Rule("note_file", ((tc.Var("x"), "reads", tc.Var("t")),), lambda b, m: [tc.Claim(tc.Ref("task:note"), "file", b["t"])])
    fired = []
    strict = Rule("parse_note", ((tc.Var("x"), "reads", tc.Var("t")),), lambda b, m: fired.append(b["t"]) or [tc.Claim(tc.Ref("task:note"), "parsed", b["t"])], established_only=True)
    t1 = integrate(mind, reads(1, "task-71008.txt"), corroboration=policy)
    d1 = think(mind, [note_file, strict], since=t1, corroboration=policy)
    (derived,) = d1.added
    assert derived.claim.predicate == "file" and policy.tentative(mind, derived.id) and fired == []
    t2 = integrate(mind, reads(2, "task-71008.txt"), corroboration=policy)
    assert {r.claim.predicate for r in t2.established} == {"reads", "file"}  # the derivation is established with its premise
    d2 = think(mind, [note_file, strict], since=t2, corroboration=policy)
    assert fired == ["task-71008.txt"] and [r.claim.predicate for r in d2.added] == ["parsed"]
    assert not policy.tentative(mind, d2.added[0].id)


def test_low_confidence_needs_confirmation_by_an_outcome():
    mind, policy = tc.Store(), Corroboration(k=2, min_confidence=0.5)
    (rec,) = integrate(mind, reads(1, "Submit", conf=0.3), corroboration=policy).added
    integrate(mind, reads(2, "Submit", conf=0.3), corroboration=policy)
    integrate(mind, reads(3, "Submit", conf=0.3), corroboration=policy)
    assert policy.tentative(mind, rec.id)
    confirmed = policy.confirm(mind, rec.id)
    assert [r.id for r in confirmed.established] == [rec.id] and not policy.tentative(mind, rec.id)


def test_ungoverned_predicates_and_no_policy_behave_as_before():
    mind, policy = tc.Store(), Corroboration(k=2, predicates=frozenset({"reads"}))
    t = integrate(mind, frame(1, seen("control:a", "label", "Staff number")), corroboration=policy)
    assert not policy.tentative(mind, t.added[0].id)
    plain = tc.Store()
    assert integrate(plain, reads(1, "x")).added and not integrate(plain, reads(2, "x")).established
