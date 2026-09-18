import enum

import pytest

import tensacode as tc
from tensacode.runtime import FunctionImplementation, Output


class Color(enum.Enum):
    red = "red"
    blue = "blue"


def impl(name, fn, **kw):
    return FunctionImplementation("classify", name, "1", fn, **kw)


def test_nothing_bound_means_unknown_not_a_model_call():
    out = tc.classify("anything", Color)
    assert isinstance(out, tc.Unknown) and out.reason == "no_implementation"


def test_unknown_and_verdict_have_no_truth_value():
    with pytest.raises(TypeError):
        bool(tc.Unknown("x"))
    with pytest.raises(TypeError):
        if tc.Verdict("unknown"):
            pass


def test_cascade_escalates_on_abstention_and_records_why():
    rt = tc.Runtime([impl("cheap", lambda r: tc.Unknown("below_threshold")), impl("better", lambda r: Color.blue)])
    with tc.use(rt):
        assert tc.classify("x", Color) is Color.blue
    (span,) = rt.trace.spans
    assert [(a.implementation, a.outcome, a.reason) for a in span.attempts] == [("cheap", "abstain", "below_threshold"), ("better", "answer", "")]
    assert span.answered_by == "better@1"


def test_final_unknown_reason_is_the_most_specific_one():
    rt = tc.Runtime([impl("a", lambda r: tc.Unknown("no_rule_matched")), impl("b", lambda r: tc.Unknown("tie_within_margin"))])
    with tc.use(rt):
        out = tc.classify("x", Color)
    assert out.reason == "tie_within_margin" and "no_rule_matched" in out.detail


def test_invalid_outputs_are_failed_attempts_not_answers():
    rt = tc.Runtime([impl("liar", lambda r: "purple"), impl("honest", lambda r: Color.red)])
    with tc.use(rt):
        assert tc.classify("x", Color) is Color.red
    assert rt.trace.spans[0].attempts[0].outcome == "invalid"


def test_crashing_backend_is_an_attempt_outcome():
    def boom(r):
        raise RuntimeError("gpu fell over")

    rt = tc.Runtime([impl("crash", boom), impl("ok", lambda r: Color.red)])
    with tc.use(rt):
        assert tc.classify("x", Color) is Color.red
    assert rt.trace.spans[0].attempts[0].outcome == "error"


def test_hard_constraints_filter_before_running():
    ran = []
    remote = impl("remote", lambda r: ran.append(1) or Color.red, traits=tc.Traits(locality="remote", egress=True))
    rt = tc.Runtime([remote], policy=tc.Policy(localities=frozenset({"in_process"})))
    with tc.use(rt):
        out = tc.classify("private text", Color)
    assert isinstance(out, tc.Unknown) and out.reason == "not_permitted" and not ran


def test_unknown_cost_is_not_zero_under_a_cost_cap():
    unmetered = impl("unmetered", lambda r: Color.red)  # Profile().usd_per_call is None
    metered = impl("metered", lambda r: Color.blue, profile=tc.Profile(usd_per_call=0.001))
    rt = tc.Runtime([unmetered, metered], budget=tc.Budget(usd=0.01))
    with tc.use(rt):
        assert tc.classify("x", Color) is Color.blue
    assert rt.trace.spans[0].attempts[0].reason == "cost unknown under a cost cap"
    assert rt.budget.spent_usd == pytest.approx(0.001)


def test_budget_attempts_bound_total_work():
    rt = tc.Runtime([impl("a", lambda r: tc.Unknown("no")), impl("b", lambda r: Color.red)], budget=tc.Budget(attempts=1))
    with tc.use(rt):
        out = tc.classify("x", Color)
    assert isinstance(out, tc.Unknown)
    assert [a.outcome for a in rt.trace.spans[0].attempts] == ["abstain", "skipped"]


def test_max_attempts_per_item():
    calls = []
    tiers = [impl(f"t{i}", lambda r, i=i: calls.append(i) or tc.Unknown("no")) for i in range(5)]
    rt = tc.Runtime(tiers, policy=tc.Policy(max_attempts=2))
    with tc.use(rt):
        tc.classify("x", Color)
    assert calls == [0, 1]


def test_cache_only_for_deterministic_implementations():
    calls = []
    det = impl("det", lambda r: calls.append("det") or Color.red)
    rt = tc.Runtime([det])
    with tc.use(rt):
        tc.classify("x", Color)
        tc.classify("x", Color)
    assert calls == ["det"] and rt.trace.spans[1].attempts[0].outcome == "cache_hit"
    calls.clear()
    nondet = impl("nondet", lambda r: calls.append("n") or Color.red, traits=tc.Traits(deterministic=False))
    with tc.use(tc.Runtime([nondet])):
        tc.classify("x", Color)
        tc.classify("x", Color)
    assert calls == ["n", "n"]


def test_batch_cascade_only_escalates_pending_items():
    seen = []

    def cheap(requests):
        seen.append(("cheap", len(requests)))
        return [Output(Color.red) if r.subject == "easy" else Output(tc.Unknown("hard")) for r in requests]

    def strong(requests):
        seen.append(("strong", len(requests)))
        return [Output(Color.blue) for _ in requests]

    rt = tc.Runtime([impl("cheap", cheap, batched=True), impl("strong", strong, batched=True)])
    with tc.use(rt):
        out = tc.classify.many(["easy", "hard", "easy", "hard"], Color)
    assert out == [Color.red, Color.blue, Color.red, Color.blue]
    assert seen == [("cheap", 4), ("strong", 2)]


def test_ordering_by_cost_puts_unknown_last():
    order = []
    a = impl("unknown-cost", lambda r: order.append("u") or tc.Unknown("x"))
    b = impl("cheap", lambda r: order.append("c") or tc.Unknown("x"), profile=tc.Profile(usd_per_call=0.01))
    with tc.use(tc.Runtime([a, b], policy=tc.Policy(order="cheapest"))):
        tc.classify("x", Color)
    assert order == ["c", "u"]


def test_choose_enforces_constraints_itself():
    options = ["refund_100", "refund_10000"]
    greedy = FunctionImplementation("choose", "greedy", "1", lambda r: "refund_10000")
    constraint = tc.Constraint("under_limit", lambda o, g: o != "refund_10000")
    with tc.use(tc.Runtime([greedy])):
        out = tc.choose(options, objective=tc.Objective("resolve", "resolve"), constraints=[constraint])
    assert out == "refund_100"  # the only feasible option; the backend was never asked


def test_choose_rejects_backend_output_outside_feasible_set():
    options = ["a", "b", "c"]
    rogue = FunctionImplementation("choose", "rogue", "1", lambda r: "c")
    with tc.use(tc.Runtime([rogue])) as rt:
        out = tc.choose(options, objective=tc.Objective("o", "o"), constraints=[tc.Constraint("not_c", lambda o, g: o != "c")])
    assert isinstance(out, tc.Unknown)
    assert rt.trace.spans[0].attempts[0].outcome == "invalid"


def test_undetermined_constraint_excludes_option():
    unsure = tc.Constraint("safe", lambda o, g: tc.Unknown("no_data") if o == "b" else True)
    with tc.use(tc.Runtime()) as rt:
        assert tc.choose(["a", "b"], objective=tc.Objective("o", "o"), constraints=[unsure]) == "a"
    assert "safe=unknown(no_data)" in rt.trace.spans[0].notes[0]


def test_scores_declare_their_kind():
    with pytest.raises(ValueError):
        tc.Score(0.9, "probability")  # a calibrated probability must say what it was calibrated on
    assert tc.Score(0.9, "similarity").kind == "similarity"


def test_trace_is_serializable():
    rt = tc.Runtime([impl("a", lambda r: Color.red)])
    with tc.use(rt):
        tc.classify("x", Color)
    line = rt.trace.to_jsonl()
    assert '"op": "classify"' in line and "Color.red" in line


def test_check_many_is_three_valued_per_item():
    verdicts = {"a": tc.Verdict("holds"), "b": tc.Verdict("fails")}
    checker = FunctionImplementation("check", "table", "1", lambda r: verdicts.get(r.subject, tc.Unknown("no_record")))
    with tc.use(tc.Runtime([checker])):
        out = tc.check.many(["a", "b", "c"])
    assert [v.status for v in out] == ["holds", "fails", "unknown"]
