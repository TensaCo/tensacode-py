"""Onboarding access requests as a procedural script (kept for comparison with the mind in access.py)."""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from datetime import date

import tensacode as tc
from tensacode.backends.builtin import IN_PROCESS

from ..browser import Browser, Control, Screen, find, label_similarity, submit

# ------------------------------------------------------------------ program


@dataclass(frozen=True)
class TicketOutcome:
    ticket: str
    status: str  # "submitted" | "sent_back" | "escalated"
    reason: str


def process_queue(ui: Browser) -> list[TicketOutcome]:
    handled: list[TicketOutcome] = []
    while ticket := next_ticket(ui.observe(), {h.ticket for h in handled}):
        ui.click(ticket)
        tid = ticket.name.split()[0]
        request = tc.parse(TicketText(tid, ui.observe().text("Ticket text")), AccessRequest)
        if isinstance(request, tc.Unknown) or request.missing:
            handled.append(send_back(ui, tid, request))
            continue
        fields = tc.choose(field_assignments(ui.observe()), objective=BEST_LABEL_FIT, constraints=FIELD_TYPES)
        if isinstance(fields, tc.Unknown):
            handled.append(TicketOutcome(tid, "escalated", f"form fields unclear: {fields.reason}"))
            continue
        fill_form(ui, fields, request)
        button = find(ui.observe(), "Submit request", roles={"button"})
        result = submit(ui, button, success="created", already_done=lambda s: listed(s, tid))
        handled.append(TicketOutcome(tid, "submitted" if result.status == "done" else "escalated", result.reason))
    return handled


# ------------------------------------------------------------ parse: ticket


@dataclass(frozen=True)
class TicketText:
    ticket: str
    text: str


@dataclass(frozen=True)
class AccessRequest:
    name: str
    employee_id: str | None
    department: str
    start: date
    systems: tuple[str, ...]
    manager: str | None

    @property
    def missing(self) -> tuple[str, ...]:
        return tuple(k for k in ("employee_id", "manager") if getattr(self, k) is None)


DEPARTMENTS = {
    "Engineering": ("engineering", "eng", "platform engineers"),
    "Data Platform": ("data platform", "data team"),
    "Finance": ("finance", "accounting"),
    "Sales": ("sales",),
    "People": ("people ops", "hr"),
}
SYSTEMS = {"VPN": ("vpn",), "GitHub": ("github",), "Salesforce": ("salesforce", "sfdc"), "Payroll": ("payroll",), "Jira": ("jira",)}
MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
YEAR = 2026

_NAME = re.compile(r"(?:access for|New starter:|onboard)\s+([A-Z][a-z]+ [A-Z][a-z]+)")
_EMP = re.compile(r"\b(?:E-?|employee #|ID )(\d{5})\b")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_MONTH_DAY = re.compile(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.? (\d{1,2})(?:st|nd|rd|th)?\b")
_SLASH = re.compile(r"\b(\d{1,2})/(\d{1,2})\b")


def _words(text: str) -> str:
    return " " + re.sub(r"[^a-z0-9 ]", " ", text.lower()) + " "


@tc.implementation("parse", name="access-ticket-rules", version="1", accepts=lambda r: isinstance(r.subject, TicketText) and r.target is AccessRequest, profile=IN_PROCESS)
def parse_ticket(request: tc.Request) -> AccessRequest | tc.Unknown:
    text = request.subject.text
    name = _NAME.search(text)
    words = _words(text)
    depts = {d for d, syns in DEPARTMENTS.items() if any(f" {s} " in words for s in syns)}
    systems = tuple(sorted(s for s, syns in SYSTEMS.items() if any(f" {w} " in words for w in syns)))
    if m := _ISO.search(text):
        start = date(int(m[1]), int(m[2]), int(m[3]))
    elif m := _MONTH_DAY.search(text):
        start = date(YEAR, MONTHS[m[1].lower()], int(m[2]))
    elif m := _SLASH.search(text):
        start = date(YEAR, int(m[1]), int(m[2]))
    else:
        return tc.Unknown("no_start_date")
    if not name or len(depts) != 1 or not systems:
        return tc.Unknown("incomplete_ticket", f"name={bool(name)} departments={sorted(depts)} systems={systems}")
    emp, email = _EMP.search(text), _EMAIL.search(text)
    return AccessRequest(name[1], f"E{emp[1]}" if emp else None, depts.pop(), start, systems, email[0].rstrip(".") if email else None)


# ------------------------------------------------- choose: which box is which

TEXT_CONCEPTS = {
    "name": ("full name", "employee name", "name of new hire"),
    "employee_id": ("employee id", "staff number", "personnel no"),
    "start_date": ("start date", "first day", "access begins"),
    "manager": ("manager email", "approver email", "reports to email"),
}


@dataclass(frozen=True)
class Assignment:
    pairs: tuple[tuple[str, Control], ...]

    def control(self, concept: str) -> Control:
        return next(c for k, c in self.pairs if k == concept)

    def __repr__(self) -> str:  # compact in traces
        return "{" + ", ".join(f"{k}→{c.name!r}" for k, c in self.pairs) + "}"


def _type_evidence(concept: str, c: Control) -> float:
    hint = c.hint.lower()
    if concept == "start_date":
        return 0.6 if "yyyy" in hint else 0.0
    if concept == "manager":
        return 0.6 if c.input_type == "email" else 0.0
    if concept == "employee_id":
        return 0.6 if re.search(r"\be\d{5}\b|letter e", hint) else 0.0
    return 0.0


def _fit(concept: str, c: Control) -> float:
    return max(label_similarity(s, c.name) for s in TEXT_CONCEPTS[concept]) + _type_evidence(concept, c)


def field_assignments(screen: Screen) -> list[Assignment]:
    boxes = [c for c in screen.controls if c.role == "textbox" and c.section == "Access request form"]
    return [Assignment(tuple(zip(TEXT_CONCEPTS, perm))) for perm in itertools.permutations(boxes, len(TEXT_CONCEPTS))]


def _compatible(a: Assignment, _: object) -> bool:
    return all(_type_evidence(k, c) > 0 or k == "name" or label_similarity(TEXT_CONCEPTS[k][0], c.name) > 0.3 for k, c in a.pairs)


FIELD_TYPES = (tc.Constraint("typed_fields_have_type_evidence_or_a_label_match", _compatible),)
BEST_LABEL_FIT = tc.Objective("best_label_fit", "Maximize label similarity plus input-type evidence", lambda a, _: round(sum(_fit(k, c) for k, c in a.pairs), 6))


# ---------------------------------------------------------------- helpers


def next_ticket(screen: Screen, handled: set[str]) -> Control | None:
    return next((c for c in screen.controls_in("Request queue") if c.role == "button" and c.name.split()[0] not in handled), None)


def listed(screen: Screen, ticket: str) -> bool:
    table = screen.table("Recent submissions")
    return bool(table and any(row[-1] == ticket for row in table.rows))


def fill_form(ui: Browser, fields: Assignment, req: AccessRequest) -> None:
    date_hint = fields.control("start_date").hint
    start = req.start.isoformat() if "YYYY-MM-DD" in date_hint else req.start.strftime("%m/%d/%Y")
    for concept, value in (("name", req.name), ("employee_id", req.employee_id), ("start_date", start), ("manager", req.manager)):
        ui.fill(fields.control(concept), value)
    screen = ui.observe()
    combo = next(c for c in screen.controls if c.role == "combobox")
    ui.click(combo)
    option = find(ui.observe(), req.department, roles={"option"})
    ui.click(option)
    for box in (c for c in ui.observe().controls if c.role == "checkbox"):
        if (box.name in req.systems) != bool(box.checked):
            ui.click(box)


def send_back(ui: Browser, tid: str, why: AccessRequest | tc.Unknown) -> TicketOutcome:
    reason = f"missing {', '.join(why.missing)}" if isinstance(why, AccessRequest) else f"{why.reason}: {why.detail}"
    ui.click(find(ui.observe(), "Send back for more info", roles={"button"}))
    ui.click(find(ui.observe(), "Send back", roles={"button"}))
    return TicketOutcome(tid, "sent_back", reason)


BINDINGS = [parse_ticket]
