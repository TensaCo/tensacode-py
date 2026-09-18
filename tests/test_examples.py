"""Smoke tests: the example programs run offline with deterministic bindings only."""

from datetime import datetime, timezone

import tensacode as tc
from examples.recovery.agent import credit_with_recovery
from examples.recovery.domain import CreditAccount, Episode, Limits, classify_attempt
from examples.recovery.service import Ledger
from examples.support_router import config
from examples.support_router.agent import handle
from examples.support_router.cases import CaseLog
from examples.support_router.demo import make_bank
from examples.support_router.domain import InboundEmail, Intent
from tensacode.backends.builtin import UtilityChooser

NOW = datetime(2026, 9, 16, 9, 30, tzinfo=timezone.utc)

# Tests bind a tiny deterministic intent rule instead of the learned tier: the program is unchanged.
fraud_rule = tc.implementation("classify", name="test-fraud-rule", version="1", accepts=lambda r: r.target is Intent)(
    lambda r: Intent.lost_or_stolen_card if "stolen" in r.subject else tc.Unknown("no_rule")
)


def router_runtime():
    return tc.Runtime([config.parse_email, config.KEYWORD_RULES, fraud_rule, UtilityChooser()])


def test_router_resolves_and_persists(tmp_path):
    bank, path = make_bank(), tmp_path / "cases.json"
    with tc.use(router_runtime()):
        case = handle(InboundEmail("m1", NOW, "alice@example.com", "", "my card was stolen"), bank, CaseLog(path))
    assert case.status == "resolved" and case.receipt_status == "applied"
    reloaded = CaseLog(path).get("m1")
    assert reloaded.status == "resolved" and reloaded.intent is Intent.lost_or_stolen_card


def test_router_ambiguous_and_unknown_paths_do_not_act():
    bank = make_bank()
    with tc.use(router_runtime()):
        two_cards = handle(InboundEmail("m2", NOW, "bob@example.com", "", "wallet stolen"), bank, CaseLog())
        unknown = handle(InboundEmail("m3", NOW, "bob@example.com", "", "do you sponsor marathons?"), bank, CaseLog())
    assert two_cards.status == "needs_human" and "tie_within_margin" in two_cards.reason
    assert unknown.status == "needs_human" and bank.effects == []


def test_router_verifies_indeterminate_receipt_by_observation():
    bank = make_bank()
    bank.faults["m4:FreezeCard"] = "timeout_after_commit"
    with tc.use(router_runtime()):
        case = handle(InboundEmail("m4", NOW, "alice@example.com", "", "card stolen"), bank, CaseLog())
    assert (case.receipt_status, case.verification, case.status) == ("indeterminate", "holds", "resolved")


def test_recovery_never_duplicates_a_non_idempotent_effect_when_uncertain():
    ledger = Ledger(["timeout_after_commit", "ok"], honors_keys=False, observe_script=[False])
    with tc.use(tc.Runtime([classify_attempt, UtilityChooser()])):
        result = credit_with_recovery(Episode(CreditAccount("a", 1, "r"), Limits()), executor=ledger, observe=ledger.observe, key="k", sleep=lambda s: None)
    assert result.status == "escalated" and len(ledger.applied) == 1 and result.invocations == 1


def test_knowledge_demo_runs(capsys):
    from examples.knowledge.demo import main

    main()
    out = capsys.readouterr().out
    assert "status_at(PRN-3, 10:02)" in out and "contested" in out and "round trip equal" in out


def test_context_demo_runs(capsys):
    from examples.context_select.demo import main

    main()
    out = capsys.readouterr().out
    assert "near-duplicate of 'tkt-881'" in out and "required_evidence_exceeds_budget" in out
