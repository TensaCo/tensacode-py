"""Card-support inbox, as a mind: learned intent perception, policy-bound actions across two app views.

Intent comes from rules -> a TF-IDF classifier trained on Banking77 (abstains below a
validation-chosen threshold). No model call anywhere. Unknown intent goes to a human.
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path

import tensacode as tc
from tensacode.cognition import Fragment, Rule

from ...support_router import config as router_config
from ...support_router.domain import Intent
from ..browser import PageOutcome, SubmitAttempt
from ..mind import BY_PRIORITY, Enter, Finish, MindSpec, Press, Wait, controls, knowledge, objects, one, order

V = tc.Var
ME = tc.Ref("agent:self")
ATTENTION = tc.Ref("scope:attention")
FRAUD = {Intent.lost_or_stolen_card, Intent.compromised_card, Intent.card_payment_not_recognised}
ARTICLES = {
    Intent.card_arrival: "Where is my card?", Intent.card_delivery_estimate: "Where is my card?", Intent.pin_blocked: "Unblock your PIN",
    Intent.terminate_account: "Close your account", Intent.apple_pay_or_google_pay: "Mobile wallets", Intent.card_acceptance: "Where cards work",
    Intent.visa_or_mastercard: "Card network", Intent.declined_card_payment: "Declined payments",
}
WEB_DATA = Path(__file__).parents[1] / "web" / "data" / "inbox.json"


def data_dir() -> Path:
    return Path(os.environ.get("TENSACODE_BANKING77_DIR", "data"))


def ensure_inbox_data() -> None:
    """The inbox app shows Banking77 *test* messages; the classifier trains on the train split only."""
    if WEB_DATA.exists():
        return
    rows = router_config.load_banking77(data_dir() / "banking77_test.csv")
    WEB_DATA.parent.mkdir(parents=True, exist_ok=True)
    WEB_DATA.write_text(json.dumps([{"text": t, "label": y.value} for t, y in rows]))


@lru_cache(maxsize=1)
def bindings() -> list:
    ensure_inbox_data()
    learned, _ = router_config.learned_classifier(data_dir() / "banking77_train.csv")
    return [router_config.KEYWORD_RULES, learned]


# ------------------------------------------------------ spontaneous thoughts


def _attend(b, mind):
    msg = tc.Ref(f"message:{b['label'].split()[1]}")
    yield Fragment(tc.Ref("obs:inbox"), ((tc.Claim(ME, "reading", msg, scope=ATTENTION), None),), snapshot_of=ATTENTION, method="attend")
    if "(handled)" in b["label"]:
        yield knowledge([(tc.Claim(msg, "handled", True), None)], "obs:inbox", "read")


def _read_body(b, mind):
    msg = b["msg"]
    if not b["x"].id.startswith("text:Message body#"):
        return
    intent = tc.classify(b["text"], Intent)
    sender = next((re.sub(r"^From:\s*", "", r.claim.object) for r in mind.claims(predicate="reads") if r.claim.subject.id.startswith("text:Sender#")), None)
    claims = [(tc.Claim(msg, "intent", intent.value if isinstance(intent, Intent) else f"unknown:{intent.reason}"), None)]
    if sender:
        claims.append((tc.Claim(msg, "from", sender), None))
    yield knowledge(claims, f"doc:{msg.id}", "intent-cascade")


def _read_cards(b, mind):
    cells = dict(b["cells"])
    customer = next((m[1] for r in mind.claims(predicate="reads") if (m := re.match(r"Cards for (\S+)", r.claim.object))), None)
    if customer and "Last 4" in cells:
        scope = tc.Ref(f"scope:cards:{customer}:{cells['Last 4']}")
        yield Fragment(tc.Ref("obs:card-admin"), ((tc.Claim(tc.Ref(f"customer:{customer}"), "card_status", (cells["Last 4"], cells["Status"]), scope=scope), None),), snapshot_of=scope, method="read-table")


def _feedback(b, mind):
    outcome = tc.classify(SubmitAttempt((b["text"],), "frozen"), PageOutcome)
    msg = one(mind, ME, "reading")
    if msg is not None and isinstance(outcome, PageOutcome) and outcome is not PageOutcome.no_feedback and objects(mind, msg, "freeze_attempt"):
        yield knowledge([(tc.Claim(msg, "freeze_feedback", (b["a"].id, outcome.value)), None)], b["a"].id, "page-feedback-rules@1")


RULES = [
    Rule("attend_to_selected_message", ((V("b"), "in", "Messages"), (V("b"), "current", True), (V("b"), "label", V("label"))), _attend),
    Rule("read_message_and_classify_intent", ((ME, "reading", V("msg")), (V("x"), "reads", V("text"))), _read_body),
    Rule("read_card_table", ((V("r"), "in_table", "Cards"), (V("r"), "cells", V("cells"))), _read_cards),
    Rule("announcement_feedback", ((V("a"), "announces", V("text")),), _feedback),
]


# ------------------------------------------------------------ deliberation


def _tab(mind, label):
    return next((t for t in controls(mind, role="tab") if one(mind, t, "label") == label), None)


def _pick(mind, combo_label: str, option: str, why: str, priority: float) -> list[object] | None:
    combo = [c for c in controls(mind, role="combobox") if one(mind, c, "label") == combo_label]
    if not combo or one(mind, combo[0], "shows") == option:
        return None
    options = controls(mind, role="option", label=option)
    return [Press(options[0], why, priority=priority) if options else Press(combo[0], f"open {combo_label.lower()}", priority=priority - 1)]


def intentions(mind: tc.Store) -> list[object]:
    dialog = one(mind, tc.Ref("ui:dialog"), "shows") or ""
    msg = one(mind, ME, "reading")
    plan = one(mind, msg, "plan") if msg else None
    if dialog.startswith("Freeze card ending"):
        want = "Freeze" if plan and plan.startswith("freeze:") and plan.split(":")[1] in dialog else "Cancel"
        return [Press(b, f"dialog: {want}", priority=100) for b in controls(mind, role="button", label=want)]
    inbox_tab, admin_tab = _tab(mind, "Inbox"), _tab(mind, "Card admin")
    in_inbox = inbox_tab is not None and one(mind, inbox_tab, "current")
    if in_inbox:
        items = controls(mind, role="button", section="Messages")
        todo = [b for b in items if "(handled)" not in (one(mind, b, "label") or "") and not one(mind, tc.Ref(f"message:{one(mind, b, 'label').split()[1]}"), "handled")]
        if not items:
            return [Wait(25, "waiting for the inbox to load", priority=1)]
        if msg is None or one(mind, msg, "handled"):
            if not todo:
                return [Finish(f"{len(items)} messages handled", priority=90)]
            return [Press(b, f"read {one(mind, b, 'label')}", priority=80 + order(b, mind)) for b in todo]
    if msg is None:
        return [Press(inbox_tab, "back to inbox", priority=70)] if inbox_tab else [Wait(20, "loading", priority=1)]
    intent, sender = one(mind, msg, "intent"), one(mind, msg, "from")
    if intent is None:
        return [Wait(20, "classifying intent", priority=1)] if in_inbox else [Press(inbox_tab, "back to inbox", priority=70)]
    known = Intent(intent) if not intent.startswith("unknown:") else None

    if known in FRAUD:
        cards = objects(mind, tc.Ref(f"customer:{sender}"), "card_status")
        if not cards:  # look the customer up before deciding
            if not (admin_tab and one(mind, admin_tab, "current")):
                return [Press(admin_tab, f"{intent}: look up {sender}'s cards", priority=60)]
            box = controls(mind, role="textbox", label="Customer email")
            if box and one(mind, box[0], "value") != sender:
                return [Enter(box[0], sender, f"search {sender}", priority=58)]
            return [Press(b, "search", priority=57) for b in controls(mind, role="button", label="Search")]
        active = [last4 for last4, status in cards if status == "Active"]
        frozen_by_me = [last4 for last4 in objects(mind, msg, "freeze_attempt") if (last4, "Frozen") in cards]
        if len(active) == 1 and not frozen_by_me and plan is None:
            return [Press(b, f"freeze {active[0]}: sole active card, fraud risk", priority=55, records=(tc.Claim(msg, "plan", f"freeze:{active[0]}"), tc.Claim(msg, "freeze_attempt", active[0]))) for b in controls(mind, role="button", label=f"Freeze card ending {active[0]}")] or [Wait(20, "waiting for card list", priority=1)]
        if plan and plan.startswith("freeze:") and not frozen_by_me:
            last4 = plan.split(":")[1]
            feedback = [v for _, v in objects(mind, msg, "freeze_feedback")]
            if feedback and feedback[-1] == "transient_failure" and len(feedback) < 3:
                return [Press(b, f"retry freeze {last4} after 503", priority=54) for b in controls(mind, role="button", label=f"Freeze card ending {last4}")]
            return [Wait(25, "waiting for freeze", priority=2)]
        label = "Fraud – card frozen" if frozen_by_me else "Needs human"
    elif known in ARTICLES:
        label = None
    else:
        label = "Needs human"

    if not in_inbox:
        return [Press(inbox_tab, "back to inbox", priority=50)]
    if label is None:
        article = ARTICLES[known]
        return _pick(mind, "Help article", article, f"{intent} → article '{article}'", 45) or [Press(b, f"reply with '{article}'", priority=40) for b in controls(mind, role="button", label="Send reply")]
    return _pick(mind, "Label", label, f"{intent} → label '{label}'", 45) or [Press(b, f"apply label '{label}'", priority=40) for b in controls(mind, role="button", label="Apply label")]


SPEC = MindSpec("inbox", RULES, intentions, BY_PRIORITY, max_cycles=250)
