"""Onboarding access requests, as a mind.

Perception gives the screen as claims. Spontaneous thoughts (rules) read tickets into
knowledge, guess what each form field means, and interpret page announcements.
Deliberation picks one intention per cycle; constraints keep submits safe.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass

import tensacode as tc
from tensacode.cognition import Fragment, Rule

from ..browser import PageOutcome, SubmitAttempt, label_similarity
from ..mind import BY_PRIORITY, Escalate, Finish, MindSpec, Note, Press, Enter, Wait, controls, knowledge, objects, one, order, subjects
from .access_script import TEXT_CONCEPTS, AccessRequest, TicketText, _type_evidence, parse_ticket

ME = tc.Ref("agent:self")
QUEUE, FORM, TICKET_TEXT = "Request queue", "Access request form", "Ticket text"
V = tc.Var


def ticket_ref(label: str) -> tc.Ref:
    return tc.Ref(f"ticket:{label.split()[0]}")


# ------------------------------------------------------ spontaneous thoughts


ATTENTION = tc.Ref("scope:attention")


def _working_on(b, mind):
    # attention is its own scope: it persists when the queue is hidden (e.g. by a dialog) and is
    # replaced, not accumulated, when the focus moves
    claim = tc.Claim(ME, "working_on", ticket_ref(b["label"]), scope=ATTENTION)
    yield Fragment(tc.Ref("obs:queue"), ((claim, None),), snapshot_of=ATTENTION, method="attend")


def _read_ticket(b, mind):
    if not b["x"].id.startswith(f"text:{TICKET_TEXT}#"):
        return
    ticket = b["t"]
    req = tc.parse(TicketText(ticket.id, b["text"]), AccessRequest)
    if isinstance(req, tc.Unknown):
        yield knowledge([(tc.Claim(ticket, "unreadable", req.reason), None)], f"doc:{ticket.id}", "access-ticket-rules@1")
        return
    facts = {"name": req.name, "employee_id": req.employee_id, "department": req.department, "start": req.start, "manager": req.manager}
    claims = [(tc.Claim(ticket, k, v), None) for k, v in facts.items() if v is not None]
    claims += [(tc.Claim(ticket, "needs_system", s), None) for s in req.systems]
    claims += [(tc.Claim(ticket, "missing", m), None) for m in req.missing]
    yield knowledge(claims, f"doc:{ticket.id}", "access-ticket-rules@1")


def _queue_status(b, mind):
    words = b["label"].split()
    if len(words) == 2 and words[1] in ("Submitted", "Returned"):
        yield knowledge([(tc.Claim(ticket_ref(b["label"]), "status", words[1].lower()), None)], "obs:queue", "read")


def _may_mean(b, mind):
    f, label = b["f"], b["label"]
    hint = one(mind, f, "hint", "")
    ctrl = mind.get(f)
    for concept, synonyms in TEXT_CONCEPTS.items():
        score = max(label_similarity(s, label) for s in synonyms) + _type_evidence(concept, ctrl)
        if score >= 0.3:
            yield tc.Claim(f, "may_mean", concept), tc.Score(round(score, 3), "relevance")  # label similarity + type evidence
    _ = hint


def _signal(b, mind):
    outcome = tc.classify(SubmitAttempt((b["text"],), "created"), PageOutcome)
    ticket = one(mind, ME, "working_on")
    if isinstance(outcome, PageOutcome) and outcome is not PageOutcome.no_feedback and ticket is not None:
        yield knowledge([(tc.Claim(ticket, "feedback", (b["a"].id, outcome.value)), None)], b["a"].id, "page-feedback-rules@1")


def _listed(b, mind):
    ticket = dict(b["cells"]).get("Ticket")
    if ticket:
        yield knowledge([(tc.Claim(tc.Ref(f"ticket:{ticket}"), "listed_in_recent", True), None)], b["r"].id, "read")


RULES = [
    Rule("working_on_current_ticket", ((V("b"), "current", True), (V("b"), "in", QUEUE), (V("b"), "label", V("label"))), _working_on),
    Rule("read_ticket_text", ((ME, "working_on", V("t")), (V("x"), "reads", V("text"))), _read_ticket),
    Rule("queue_status", ((V("b"), "in", QUEUE), (V("b"), "label", V("label"))), _queue_status),
    Rule("field_may_mean", ((V("f"), "in", FORM), (V("f"), "is_a", "textbox"), (V("f"), "label", V("label"))), _may_mean),
    Rule("announcement_signals", ((V("a"), "announces", V("text")),), _signal),
    Rule("listed_in_recent", ((V("r"), "in_table", "Recent submissions"), (V("r"), "cells", V("cells"))), _listed),
]


# ----------------------------------------------------------- deliberation


@dataclass(frozen=True)
class FieldPlan:
    pairs: tuple[tuple[str, tc.Ref], ...]
    fit: float

    def __repr__(self) -> str:
        return "{" + ", ".join(f"{k}→{r.id.split('/')[-1]}" for k, r in self.pairs) + "}"


def _field_plans(mind: tc.Store) -> list[FieldPlan]:
    boxes = controls(mind, role="textbox", section=FORM)
    score = {(r.claim.object, r.claim.subject): r.evidence[0].confidence.value for r in mind.claims(predicate="may_mean")}
    plans = []
    for perm in itertools.permutations(boxes, len(TEXT_CONCEPTS)):
        pairs = tuple(zip(TEXT_CONCEPTS, perm))
        if all((k, f) in score for k, f in pairs):
            plans.append(FieldPlan(pairs, round(sum(score[p] for p in pairs), 6)))
    return plans


def _expected(mind: tc.Store, ticket: tc.Ref, concept: str, field: tc.Ref) -> str | None:
    if concept == "start_date":
        start = one(mind, ticket, "start")
        return None if start is None else (start.isoformat() if "YYYY-MM-DD" in one(mind, field, "hint", "") else start.strftime("%m/%d/%Y"))
    return one(mind, ticket, concept)


MAX_UNSAFE_SUBMITS, MAX_SUBMITS = 3, 6  # attempts that may have had an effect / all attempts


def _risky_attempts(mind: tc.Store, ticket: tc.Ref) -> int:
    """Attempts that may have saved something. A 503 that says nothing was saved does not count."""
    feedback = [v for _, v in objects(mind, ticket, "feedback")]
    return len(objects(mind, ticket, "attempt")) - feedback.count("transient_failure")


def _handled(mind: tc.Store, ticket: tc.Ref) -> bool:
    feedback = [v for _, v in objects(mind, ticket, "feedback")]
    return (
        one(mind, ticket, "status") in ("submitted", "returned")
        or "confirmed" in feedback
        or (one(mind, ticket, "listed_in_recent") and "unconfirmed" in feedback)
        or one(mind, ticket, "gave_up") is not None
    )


def intentions(mind: tc.Store) -> list[object]:
    queue = controls(mind, role="button", section=QUEUE)
    ticket = one(mind, ME, "working_on")
    dialog = one(mind, tc.Ref("ui:dialog"), "shows")
    if dialog:
        wanted = "Send back" if ticket is not None and (objects(mind, ticket, "missing") or one(mind, ticket, "unreadable")) else "Cancel"
        return [Press(b, f"confirm dialog: {wanted}", priority=100) for b in controls(mind, role="button", label=wanted)]
    if ticket is None or _handled(mind, ticket):
        todo = [b for b in queue if not _handled(mind, ticket_ref(one(mind, b, "label")))]
        if not todo:
            gave_up = [r.claim.subject.id for r in mind.claims(predicate="gave_up")]
            if gave_up:
                return [Escalate(f"{len(queue) - len(gave_up)} of {len(queue)} tickets handled; gave up on {', '.join(gave_up)}: {'; '.join(r.claim.object for r in mind.claims(predicate='gave_up'))}")]
            return [Finish(f"all {len(queue)} tickets handled", priority=90)]
        return [Press(b, f"open {one(mind, b, 'label').split()[0]}", priority=80 + order(b, mind)) for b in todo]
    if one(mind, ticket, "name") is None and one(mind, ticket, "unreadable") is None:
        return [Wait(20, "reading the ticket", priority=1)]
    if objects(mind, ticket, "missing") or one(mind, ticket, "unreadable"):
        why = ", ".join(objects(mind, ticket, "missing")) or one(mind, ticket, "unreadable")
        return [Press(b, f"send back {ticket.id}: {why}", priority=70) for b in controls(mind, role="button", label="Send back for more info")]

    meaning = {r.claim.subject: r.claim.object for r in mind.claims(predicate="means")}
    if not meaning:
        plan = tc.choose(_field_plans(mind), objective=tc.Objective("fit", "best label+type fit", lambda p, m: p.fit))
        if isinstance(plan, tc.Unknown):
            return [Escalate(f"form fields unclear: {plan.reason}")]
        premises = tuple(r.id for r in mind.claims(predicate="may_mean") if (r.claim.object, r.claim.subject) in plan.pairs)
        return [Note(tuple(tc.Claim(f, "means", k) for k, f in plan.pairs), premises, f"fields: {plan!r}", priority=60)]

    todo: list[object] = []
    for field, concept in meaning.items():
        want = _expected(mind, ticket, concept, field)
        if want is not None and one(mind, field, "value") != want:
            todo.append(Enter(field, want, f"{concept} := {want}", priority=50 + order(field, mind)))
    dept = one(mind, ticket, "department")
    combo = controls(mind, role="combobox", section=FORM)
    if combo and one(mind, combo[0], "shows") != dept:
        options = controls(mind, role="option", label=dept)
        todo.append(Press(options[0], f"department := {dept}", priority=45) if options else Press(combo[0], "open department list", priority=44))
    needed = set(objects(mind, ticket, "needs_system"))
    for box in controls(mind, role="checkbox", section=FORM):
        if bool(one(mind, box, "checked")) != (one(mind, box, "label") in needed):
            todo.append(Press(box, f"toggle {one(mind, box, 'label')}", priority=40 + order(box, mind)))
    if todo:
        return todo

    attempts = len(objects(mind, ticket, "attempt"))
    feedback = [v for _, v in objects(mind, ticket, "feedback")]
    if attempts > len(feedback):
        return [Wait(25, "waiting for the page to answer", priority=5)]
    if attempts >= MAX_SUBMITS or _risky_attempts(mind, ticket) >= MAX_UNSAFE_SUBMITS or (attempts and one(mind, ticket, "listed_in_recent")):
        why = f"{attempts} submits without confirmation ({', '.join(feedback)})"
        return [Note((tc.Claim(ticket, "gave_up", why),), (), f"give up on {ticket.id}: {why}", priority=35)]
    submit = controls(mind, role="button", label="Submit request")
    return [Press(b, f"submit {ticket.id} (attempt {attempts + 1})", records=(tc.Claim(ticket, "attempt", attempts + 1),), priority=30) for b in submit]


def _safe_submit(i: object, mind: tc.Store) -> bool:
    if not (isinstance(i, Press) and i.records):
        return True
    ticket = i.records[0].subject
    return i.records[0].object <= MAX_SUBMITS and _risky_attempts(mind, ticket) < MAX_UNSAFE_SUBMITS and not one(mind, ticket, "listed_in_recent")


SPEC = MindSpec(
    "access",
    RULES,
    intentions,
    BY_PRIORITY,
    constraints=(tc.Constraint("at_most_3_effectful_submits_and_never_resubmit_a_listed_request", _safe_submit),),
    max_cycles=300,
)
BINDINGS = [parse_ticket]
_ = subjects
