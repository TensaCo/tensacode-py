from dataclasses import dataclass

import tensacode as tc


@tc.action(effect="external", idempotent=True)
@dataclass(frozen=True)
class Ping:
    host: str


@tc.action(effect="read", idempotent=True)
@dataclass(frozen=True)
class Look:
    host: str


@dataclass(frozen=True)
class NotAnAction:
    x: int


class Recorder:
    def __init__(self, fail=()):
        self.done, self.fail = [], set(fail)

    def execute(self, act, *, key):
        if act.host in self.fail:
            raise TimeoutError("lost reply")
        self.done.append((act, key))
        return tc.Receipt(act, "applied", idempotency_key=key)


def test_invoke_rejects_only_what_it_cannot_execute():
    """No permission gating: an unregistered action, or a write with no idempotency key, is all that is refused."""
    ex = Recorder()
    assert tc.invoke(NotAnAction(1), executor=ex, key="k").status == "rejected"
    assert tc.invoke(Ping("a"), executor=ex, key=None).status == "rejected"
    assert tc.invoke(Look("a"), executor=ex, key=None).status == "applied"
    assert tc.invoke(Ping("a"), executor=ex, key="k").status == "applied"
    assert [a for a, _ in ex.done] == [Look("a"), Ping("a")]


def test_exception_after_dispatch_is_indeterminate_not_failed():
    receipt = tc.invoke(Ping("flaky"), executor=Recorder(fail={"flaky"}), key="k")
    assert receipt.status == "indeterminate" and not receipt.retryable


def test_verify_uses_observation_not_the_receipt():
    applied = tc.Receipt(Ping("a"), "applied")
    assert tc.verify(applied, observe=lambda: {"a": False}, expect=lambda obs: obs["a"]).status == "fails"
    assert tc.verify(applied, observe=lambda: tc.Unknown("down"), expect=lambda obs: True).status == "unknown"
    lost = tc.Receipt(Ping("a"), "indeterminate")
    assert tc.verify(lost, observe=lambda: {"a": True}, expect=lambda obs: obs["a"]).holds


def test_plan_structure_is_checked_and_dependencies_respected():
    bad = tc.Plan((tc.Step("1", Ping("a")), tc.Step("2", NotAnAction(1), needs=("1",))))
    assert isinstance(tc.plan_order(bad), tc.Verdict)
    dangling = tc.Plan((tc.Step("1", Ping("a"), needs=("nope",)),))
    assert tc.plan_order(dangling).reasons == ("1: depends on unknown step nope",)
    cyclic = tc.Plan((tc.Step("1", Ping("a"), needs=("2",)), tc.Step("2", Ping("b"), needs=("1",))))
    assert tc.plan_order(cyclic).reasons == ("dependency cycle",)
    plan = tc.Plan((tc.Step("notify", Ping("b"), needs=("fix",)), tc.Step("fix", Ping("flaky"))))
    runnable = tc.plan_order(plan)
    assert isinstance(runnable, tc.RunnablePlan) and runnable.order == ("fix", "notify")
    ex = Recorder(fail={"flaky"})
    results = tc.run_plan(runnable, executor=ex, key_prefix="p1")
    assert results["fix"].status == "indeterminate"
    assert isinstance(results["notify"], tc.Unknown) and ex.done == []


def test_a_plan_can_run_straight_from_the_plan():
    ex = Recorder()
    results = tc.run_plan(tc.Plan((tc.Step("a", Ping("a")), tc.Step("b", Ping("b"), needs=("a",)))), executor=ex, key_prefix="p2")
    assert [r.status for r in results.values()] == ["applied", "applied"]
    assert [a.host for a, _ in ex.done] == ["a", "b"]


def test_pack_never_drops_required_evidence():
    items = [(f"doc{i}", tc.Score(10 - i, "relevance")) for i in range(10)]
    packed = tc.pack(items, budget=12, cost=lambda s: 4, required=["doc9"], key=lambda s: s)
    assert packed.items[0] == "doc9" and packed.used <= 12 and len(packed.items) == 3
    assert {s for s, _ in packed.dropped} == {f"doc{i}" for i in range(2, 9)}
    assert isinstance(tc.pack(items, budget=3, cost=lambda s: 4, required=["doc9"]), tc.Unknown)


def test_dedupe_keeps_required_and_reports_duplicates():
    ranked = [("the printer is offline on floor two", tc.Score(2, "relevance")), ("FW: the printer is offline on floor two", tc.Score(1, "relevance"))]
    kept, dropped = tc.dedupe(ranked, similarity=tc.shingle_similarity, threshold=0.5, key=lambda s: s[:10])
    assert len(kept) == 1 and "near-duplicate" in dropped[0][1]
    kept, dropped = tc.dedupe(ranked, similarity=tc.shingle_similarity, threshold=0.5, keep=lambda s: s.startswith("FW"))
    assert len(kept) == 2 and not dropped
