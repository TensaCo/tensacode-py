"""The application program. Runs against ``tensacode`` (the prototype), not legacy ``tensacode``."""

from __future__ import annotations

import tensacode as tc

from .bank import Bank
from .cases import Case, CaseLog
from .domain import RESOLVE_SAFELY, SUPPORT_POLICY, InboundEmail, Intent, Situation, SupportRequest, options


def handle(email: InboundEmail, bank: Bank, cases: CaseLog) -> Case:
    request = tc.parse(email, SupportRequest)
    if isinstance(request, tc.Unknown):
        return cases.needs_human(email, "could not parse message", request)

    intent = tc.classify(request.text, Intent)
    if isinstance(intent, tc.Unknown):
        return cases.needs_human(email, "intent unknown", intent)

    account = bank.find_account(request.sender)
    situation = Situation(intent, request, account, cases.recent_actions(request.sender, hours=24))
    action = tc.choose(options(intent, request, account), objective=RESOLVE_SAFELY, given=situation, constraints=SUPPORT_POLICY)
    if isinstance(action, tc.Unknown):
        return cases.needs_human(email, "no single safe action", action, intent=intent)

    key = f"{email.id}:{type(action).__name__}"
    cases.begin(email, intent, action, key)  # durable intent before any effect
    receipt = tc.invoke(action, executor=bank, key=key)
    verdict = tc.verify(receipt, observe=lambda: bank.observe(action), expect=action.achieved)
    return cases.finish(email, receipt, verdict)
