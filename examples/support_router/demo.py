"""Run the support router over hand-written scenarios and print cases plus traces.

    python -m examples.support_router.demo --train path/to/banking77_train.csv [--model Qwen/Qwen2.5-7B-Instruct]

The scenarios are illustrations, not an evaluation; see eval/ for measured results.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import tensorcode as tc

from . import config
from .agent import handle
from .bank import Bank
from .cases import CaseLog
from .domain import Account, Card, InboundEmail


def make_bank() -> Bank:
    accounts = [
        Account("cus-alice", "alice@example.com", (Card("card-a1", "4242", "active"),)),
        Account("cus-bob", "bob@example.com", (Card("card-b1", "1111", "active"), Card("card-b2", "2222", "active"))),
        Account("cus-carol", "carol@example.com", (Card("card-c1", "3333", "frozen"),)),
        Account("cus-dan", "dan@example.com", (Card("card-d1", "5555", "active"),)),
        Account("cus-erin", "erin@example.com", (Card("card-e1", "7777", "active"),)),
    ]
    return Bank({a.customer_id: a for a in accounts})


NOW = datetime(2026, 9, 16, 9, 30, tzinfo=timezone.utc)
SCENARIOS = [
    ("happy path", InboundEmail("m-001", NOW, "alice@example.com", "Lost card", "I can't find my card anywhere, I think I lost it on the train. Please block it.")),
    ("ambiguous action (two cards)", InboundEmail("m-002", NOW, "bob@example.com", "", "My wallet was stolen with my cards in it.")),
    ("disambiguated by last4", InboundEmail("m-003", NOW, "bob@example.com", "", "My wallet was stolen, including the card ending in 2222.")),
    ("unknown intent", InboundEmail("m-004", NOW, "alice@example.com", "Hello", "Does your company sponsor the city marathon this year?")),
    ("learned tier unsure", InboundEmail("m-009", NOW, "alice@example.com", "", "the shop says my card isn't accepted there, what gives?")),
    ("constraint excludes freeze", InboundEmail("m-005", NOW, "carol@example.com", "", "Somebody stole my card yesterday.")),
    ("executor unavailable", InboundEmail("m-006", NOW, "dan@example.com", "", "My card was stolen this morning, please freeze it.")),
    ("reply lost after commit", InboundEmail("m-007", NOW, "erin@example.com", "", "My card was stolen this morning, please freeze it.")),
    ("unparseable", InboundEmail("m-008", NOW, "alice@example.com", "", "   ")),
]
FAULTS = {"m-006:FreezeCard": "unavailable", "m-007:FreezeCard": "timeout_after_commit"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--model", default=None, help="optional local instruction model for classify escalation")
    args = ap.parse_args()

    learned, report = config.learned_classifier(args.train)
    print(f"learned tier: threshold={report.threshold:.3f} ({report.basis}); validation coverage {report.validation_coverage:.1%}\n")
    model = config.local_model_classifier(args.model) if args.model else None
    runtime = tc.Runtime(config.bindings(learned=learned, model=model), policy=config.WITH_LOCAL_MODEL if model else config.LOCAL_ONLY)

    bank, cases = make_bank(), CaseLog()
    bank.faults.update(FAULTS)
    with tc.use(runtime):
        for title, email in SCENARIOS:
            mark = len(runtime.trace.spans)
            with runtime.trace.section("handle", scenario=title, message=email.id):
                case = handle(email, bank, cases)
            print(f"### {title} ({email.id})")
            print(f"case: status={case.status} intent={case.intent.value if case.intent else None} action={case.action}")
            print(f"      receipt={case.receipt_status} verification={case.verification} reason={case.reason!r}")
            print("trace:")
            print(runtime.trace.render(since=mark))
            print()
    print("effects that actually happened at the bank:")
    for kind, act in bank.effects:
        print(f"  {kind}: {act}")


if __name__ == "__main__":
    main()
