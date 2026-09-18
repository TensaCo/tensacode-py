"""Accounts-payable reconciliation, as a mind.

Every table row becomes a claim with evidence. The same proposition read from two
bank rows is one claim with two pieces of evidence, which is exactly what a duplicate
payment is. Discrepancies are concluded from the graph and filed with their evidence.
"""

from __future__ import annotations

import re

import tensacode as tc
from tensacode.cognition import Rule

from ..browser import PageOutcome, SubmitAttempt
from ..mind import BY_PRIORITY, Enter, Finish, MindSpec, Note, Press, Wait, controls, knowledge, objects, one, order

V = tc.Var
INVOICES, PAYMENTS, LEDGER = tc.Ref("list:invoices"), tc.Ref("list:payments"), tc.Ref("agent:ledger")
_REF = re.compile(r"(?:INV-?|inv#|Invoice\s+)(\d{4})", re.I)


def amount(text: str) -> float:
    return float(text.replace("$", "").replace(",", ""))


# ------------------------------------------------------ spontaneous thoughts


def _invoice_row(b, mind):
    cells = dict(b["cells"])
    if "Invoice" in cells:
        yield knowledge([(tc.Claim(tc.Ref(f"invoice:{cells['Invoice']}"), "billed", amount(cells["Amount"])), f"vendor {cells['Vendor']}")], f"row:invoices/{cells['Invoice']}", "read-table")


def _payment_row(b, mind):
    cells = dict(b["cells"])
    if "Payment" not in cells:
        return
    pay = tc.Ref(f"payment:{cells['Payment']}")
    m = _REF.search(cells["Reference"])
    claims = [(tc.Claim(pay, "amount", amount(cells["Amount"])), None), (tc.Claim(pay, "reference_text", cells["Reference"]), None)]
    if m:
        invoice = tc.Ref(f"invoice:INV-{m[1]}")
        claims += [(tc.Claim(pay, "pays", invoice), None)]
        yield knowledge([(tc.Claim(invoice, "paid", amount(cells["Amount"])), cells["Reference"])], pay.id, "read-table")
    yield knowledge(claims, pay.id, "read-table")


def _pager(b, mind):
    m = re.fullmatch(r"Page (\d+) of (\d+)", b["text"])
    lst = INVOICES if b["x"].id.startswith("text:Invoice list#") else PAYMENTS if b["x"].id.startswith("text:Payment list#") else None
    if m and lst:
        yield knowledge([(tc.Claim(lst, "seen_page", int(m[1])), None), (tc.Claim(lst, "pages", int(m[2])), None)], "obs:pager", "read")


def _filed(b, mind):
    cells = dict(b["cells"])
    if "Type" in cells:
        yield knowledge([(tc.Claim(tc.Ref(f"dispute:{cells['Type']}|{cells['Invoice / reference']}"), "filed", True), None)], b["r"].id, "read-table")


def _feedback(b, mind):
    outcome = tc.classify(SubmitAttempt((b["text"],), "Dispute filed"), PageOutcome)
    # the page answers the attempt still waiting for an answer (not whatever was filed first)
    waiting = [s for s in {r.claim.subject for r in mind.claims(predicate="attempt")} if len(objects(mind, s, "attempt")) > len(objects(mind, s, "feedback"))]
    if len(waiting) == 1 and isinstance(outcome, PageOutcome) and outcome is not PageOutcome.no_feedback:
        yield knowledge([(tc.Claim(waiting[0], "feedback", (b["a"].id, outcome.value)), None)], b["a"].id, "page-feedback-rules@1")


RULES = [
    Rule("read_invoice_row", ((V("r"), "in_table", "Invoices"), (V("r"), "cells", V("cells"))), _invoice_row),
    Rule("read_payment_row", ((V("r"), "in_table", "Payments"), (V("r"), "cells", V("cells"))), _payment_row),
    Rule("read_pager", ((V("x"), "reads", V("text")),), _pager),
    Rule("read_filed_disputes", ((V("r"), "in_table", "Filed disputes"), (V("r"), "cells", V("cells"))), _filed),
    Rule("announcement_feedback", ((V("a"), "announces", V("text")),), _feedback),
]


# -------------------------------------------------------------- conclusions


def discrepancies(mind: tc.Store) -> list[tuple[str, str, str, tuple[str, ...]]]:
    """(issue type, key, evidence sentence, premise claim ids), concluded from the claim graph."""
    found = []
    billed = {r.claim.subject: r for r in mind.claims(predicate="billed")}
    for inv, b in sorted(billed.items()):
        paid = mind.claims(inv, "paid")
        amt = b.claim.object
        if not paid:
            found.append(("Missing payment", inv.id[8:], f"{inv.id[8:]} billed ${amt:,.2f}; no payment references it in the bank feed", (b.id,)))
        for p in paid:
            payers = sorted({e.source.id[8:] for e in p.evidence})
            if p.claim.object != amt:
                found.append(("Amount mismatch", inv.id[8:], f"{inv.id[8:]} billed ${amt:,.2f}; paid ${p.claim.object:,.2f} by {', '.join(payers)}", (b.id, p.id)))
            elif len(payers) > 1:
                found.append(("Duplicate payment", inv.id[8:], f"{inv.id[8:]} billed ${amt:,.2f} and paid {len(payers)} times: {', '.join(payers)}", (b.id, p.id)))
    for r in mind.claims(predicate="pays"):
        if r.claim.object not in billed:
            pay = r.claim.subject.id[8:]
            found.append(("Unknown reference", pay, f"{pay} references {r.claim.object.id[8:]}, which is not a vendor invoice ({one(mind, r.claim.subject, 'reference_text')})", (r.id,)))
    return found


# ------------------------------------------------------------ deliberation


def _tab(mind, label):
    tabs = [t for t in controls(mind, role="tab") if one(mind, t, "label") == label]
    return tabs[0] if tabs else None


def _read_all(mind, lst: tc.Ref, tab_label: str) -> list[object] | None:
    pages, seen = one(mind, lst, "pages"), set(objects(mind, lst, "seen_page"))
    if pages is not None and len(seen) >= pages:
        return None
    tab = _tab(mind, tab_label)
    if tab and not one(mind, tab, "current"):
        return [Press(tab, f"open {tab_label}", priority=70)]
    nxt = controls(mind, role="button", label="Next page")
    return [Press(nxt[0], f"read {tab_label.lower()} page {len(seen) + 1}", priority=65)] if nxt else [Wait(25, "waiting for table", priority=1)]


def intentions(mind: tc.Store) -> list[object]:
    for lst, label in ((INVOICES, "Invoices"), (PAYMENTS, "Payments")):
        if (todo := _read_all(mind, lst, label)) is not None:
            return todo
    concluded = mind.claims(predicate="should_file")
    if not concluded and one(mind, LEDGER, "concluded") is None:
        found = discrepancies(mind)
        claims = tuple(tc.Claim(tc.Ref(f"dispute:{t}|{k}"), "should_file", text) for t, k, text, _ in found) + (tc.Claim(LEDGER, "concluded", len(found)),)
        premises = tuple(sorted({p for *_, ps in found for p in ps}))
        return [Note(claims, premises, f"concluded {len(found)} discrepancies from {len(mind.claims(predicate='billed'))} invoices", priority=60)]
    pending = [r for r in concluded if not one(mind, r.claim.subject, "filed")]
    if not pending:
        return [Finish(f"filed {len(concluded)} disputes", priority=100)]
    tab = _tab(mind, "Disputes")
    if tab and not one(mind, tab, "current"):
        return [Press(tab, "open disputes", priority=55)]
    dispute = pending[0].claim.subject
    kind, key = dispute.id[8:].split("|")
    evidence = pending[0].claim.object
    todo: list[object] = []
    combo = controls(mind, role="combobox", section="New dispute")
    if combo and one(mind, combo[0], "shows") != kind:
        options = controls(mind, role="option", label=kind)
        todo.append(Press(options[0], f"issue type := {kind}", priority=50) if options else Press(combo[0], "open issue types", priority=49))
    for label, want in (("Invoice / reference", key), ("Evidence", evidence)):
        for box in controls(mind, role="textbox", label=label):
            if one(mind, box, "value") != want:
                todo.append(Enter(box, want, f"{label.lower()} := {want[:60]}", priority=48 + order(box, mind)))
    if todo:
        return todo
    attempts = len(objects(mind, dispute, "attempt"))
    if attempts > len(objects(mind, dispute, "feedback")):
        return [Wait(25, "waiting for the ledger to answer", priority=5)]
    return [
        Press(b, f"file {kind.lower()} for {key} (attempt {attempts + 1})", priority=40, records=(tc.Claim(dispute, "attempt", attempts + 1),))
        for b in controls(mind, role="button", label="File dispute")
    ]


def _safe(i: object, mind: tc.Store) -> bool:
    return not (isinstance(i, Press) and i.records and i.records[0].predicate == "attempt") or (i.records[0].object <= 3 and not one(mind, i.records[0].subject, "filed"))


SPEC = MindSpec("recon", RULES, intentions, BY_PRIORITY, constraints=(tc.Constraint("at_most_3_attempts_and_never_refile", _safe),), max_cycles=250)
BINDINGS: list = []
