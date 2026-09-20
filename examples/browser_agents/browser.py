"""A small computer-use layer on top of tensorcode. No model calls anywhere.

* Perception: the page's accessibility-relevant structure (roles, names, values,
  hints, tables, dialogs, busy state) parsed into typed ``Screen`` values. Pixel
  perception, where needed, is done by the task (see ``tasks/chart.py``).
* Action: real mouse clicks at element coordinates and real key presses, each a
  registered action invoked through ``tc.invoke`` so it has a receipt and a span.
* Targeting: ``find`` ranks controls against a label (``tc.rank``) and abstains when
  no control matches clearly; it never clicks its best guess on a near tie.
* Submitting: ``submit`` verifies the outcome by observation and recovers within
  explicit limits (retry only when the page says nothing happened).
"""

from __future__ import annotations

import enum
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import tensorcode as tc
from tensorcode.backends.builtin import IN_PROCESS, RULES

PERCEIVE_JS = r"""
() => {
  const vis = (e) => { const r = e.getBoundingClientRect(); const s = getComputedStyle(e);
    return r.width > 0 && r.height > 0 && s.visibility !== 'hidden' && r.bottom > 0 && r.top < innerHeight && r.right > 0 && r.left < innerWidth; };
  const box = (e) => { const r = e.getBoundingClientRect(); return [Math.round(r.x), Math.round(r.y), Math.round(r.width), Math.round(r.height)]; };
  const clean = (t) => (t || '').replace(/\s+/g, ' ').trim().slice(0, 300);
  const byId = (ids) => (ids || '').split(/\s+/).map((i) => document.getElementById(i)).filter(Boolean).map((e) => e.innerText).join(' ');
  const dialog = document.querySelector('[role=dialog],[role=alertdialog]');
  const name = (e) => clean(e.getAttribute('aria-label') || byId(e.getAttribute('aria-labelledby'))
      || (e.labels && e.labels.length ? [...e.labels].map((l) => l.innerText).join(' ') : '')
      || (e.type === 'checkbox' && e.parentElement.tagName === 'LABEL' ? e.parentElement.innerText : '')
      || (['INPUT', 'TEXTAREA'].includes(e.tagName) ? '' : e.innerText) || e.title || '');
  const role = (e) => e.getAttribute('role') || (e.tagName === 'INPUT' ? (e.type === 'checkbox' ? 'checkbox' : 'textbox')
      : e.tagName === 'TEXTAREA' ? 'textbox' : e.tagName === 'A' ? 'link' : e.tagName.toLowerCase());
  // a visible point of the control that is not covered by something else (toasts, overlays)
  const hit = (e) => { const r = e.getBoundingClientRect();
    for (const [fx, fy] of [[.5,.5],[.25,.5],[.75,.5],[.5,.25],[.5,.75],[.15,.2],[.85,.8],[.15,.8],[.85,.2]]) {
      const x = r.left + r.width * fx, y = r.top + r.height * fy; const top = document.elementFromPoint(x, y);
      if (top && (top === e || e.contains(top) || (e.labels && [...e.labels].some((l) => l.contains(top))))) return [Math.round(x), Math.round(y)]; }
    return null; };
  const group = (e) => { const f = e.closest('fieldset'); return clean(f ? f.querySelector('legend')?.innerText : ''); };
  const controls = [...document.querySelectorAll('a,button,input,textarea,[role=button],[role=option],[role=tab],[role=checkbox],[role=combobox],[role=link]')]
    .filter(vis).filter((e) => !dialog || dialog.contains(e))
    .map((e) => ({ role: role(e), name: name(e), value: e.value ?? '', checked: typeof e.checked === 'boolean' ? e.checked : null,
      disabled: !!e.disabled, hint: clean([e.getAttribute('placeholder'), byId(e.getAttribute('aria-describedby'))].filter(Boolean).join(' · ')),
      point: hit(e), current: ['aria-current', 'aria-selected', 'aria-expanded'].some((a) => e.getAttribute(a) === 'true'),
      shown: e.getAttribute('role') === 'combobox' ? clean(e.innerText) : '', group: group(e), section: clean(e.parentElement?.closest('[aria-label]')?.getAttribute('aria-label')), input_type: e.type || '', box: box(e) }));
  // like a screen reader tracking live regions: remember each announcement element, so "new" means new, not new text
  const seen = (window.__tcSeen ||= { map: new WeakMap(), n: 0 });
  const seq = (e) => { if (!seen.map.has(e)) seen.map.set(e, ++seen.n); return seen.map.get(e); };
  const structured = [...document.querySelectorAll('h1,h2,h3,p,li,dt,dd,[role=status],[role=alert],[data-text]')];
  const texts = structured.filter(vis)
    .map((e) => ({ role: e.getAttribute('role') || e.tagName.toLowerCase(), text: clean(e.innerText), section: clean(e.closest('[aria-label]')?.getAttribute('aria-label')), box: box(e),
      seq: ['status', 'alert'].includes(e.getAttribute('role')) ? seq(e) : 0 }))
    .filter((t) => t.text);
  // screen-reader-style reading of remaining visible text blocks (terminal lines, labels in custom widgets)
  const region = (e) => {
    for (let a = e; a && a !== document.body; a = a.parentElement) {
      const l = (a.getAttribute('aria-label') || '').trim();
      if (l) return clean(l);
      if (a.matches('article,[role=region],[role=dialog]')) { const t = a.querySelector(':scope > header b, :scope > header h1, :scope > h1, :scope > h2'); if (t) return clean(t.innerText); }
    }
    return '';
  };
  const blocks = new Set();
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  for (let n = walker.nextNode(); n; n = walker.nextNode()) {
    if (!n.nodeValue.trim()) continue;
    let b = n.parentElement;
    while (b && getComputedStyle(b).display.startsWith('inline')) b = b.parentElement;
    if (!b || b.closest('a,button,label,[role=button],[role=option],[role=tab],h1,h2,h3,p,li,dt,dd,[role=status],[role=alert],[data-text],table,script,style,[aria-hidden=true]')) continue;
    if (structured.some((e) => b.contains(e))) continue;  // a container of already-read text would read it twice
    blocks.add(b);
  }
  const lines = [...blocks].filter(vis).map((e) => ({ role: 'text', text: e.innerText.replace(/\u00a0/g, ' ').trim().slice(0, 2000), section: region(e), box: box(e), seq: 0 })).filter((t) => t.text);
  texts.push(...lines);
  const tables = [...document.querySelectorAll('table')].filter(vis).map((t) => ({ label: t.getAttribute('aria-label') || '',
    header: [...t.querySelectorAll('thead th')].map((th) => clean(th.innerText)),
    rows: [...t.querySelectorAll('tbody tr')].map((tr) => [...tr.querySelectorAll('td')].map((td) => clean(td.innerText))) }));
  const graphics = [...document.querySelectorAll('canvas,img,[role=img]')].filter(vis).map((e) => ({ label: clean(e.getAttribute('aria-label') || e.getAttribute('alt')), box: box(e) }));
  return { url: location.href, title: document.title, controls, texts, tables, graphics,
    dialog: dialog ? clean(dialog.innerText) : '', busy: !!document.querySelector('[aria-busy=true]') };
}
"""


# ---------------------------------------------------------------- perception


@dataclass(frozen=True)
class Control:
    role: str
    name: str
    value: str
    checked: bool | None
    disabled: bool
    hint: str
    group: str
    section: str
    input_type: str
    box: tuple[int, int, int, int]
    point: tuple[int, int] | None  # an unobstructed point to click; None if fully covered
    current: bool  # aria-current / aria-selected / aria-expanded
    shown: str  # visible text of a combobox (its current selection)
    provenance: tuple = ()  # retained provider transport identity; not a semantic label


def is_computerworld_terminal_input(control) -> bool:
    """Recognize an explicit engine interaction identity, never display text."""
    return control.role == "textbox" and any(
        provenance.source == "computerworld" and provenance.locator.endswith(":terminal-input")
        for provenance in getattr(control, "provenance", ())
    )


@dataclass(frozen=True)
class Text:
    role: str
    text: str
    section: str
    box: tuple[int, int, int, int]
    seq: int  # for live announcements (role=status/alert): order of first appearance; 0 otherwise


@dataclass(frozen=True)
class Table:
    label: str
    header: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]

    def records(self) -> list[dict[str, str]]:
        return [dict(zip(self.header, row)) for row in self.rows]


@dataclass(frozen=True)
class Graphic:
    label: str
    box: tuple[int, int, int, int]


@dataclass(frozen=True)
class Screen:
    url: str
    title: str
    controls: tuple[Control, ...]
    texts: tuple[Text, ...]
    tables: tuple[Table, ...]
    dialog: str
    busy: bool
    graphics: tuple[Graphic, ...] = ()

    def text(self, section: str | None = None) -> str:
        return " ".join(t.text for t in self.texts if section is None or t.section == section)

    def alerts(self, after: int = 0) -> list[str]:
        return [t.text for t in self.texts if t.role in ("alert", "status") and t.seq > after]

    @property
    def last_alert(self) -> int:
        return max((t.seq for t in self.texts), default=0)

    def table(self, label: str) -> Table | None:
        return next((t for t in self.tables if t.label == label), None)

    def controls_in(self, section: str) -> list[Control]:
        return [c for c in self.controls if c.section == section]


def _screen(raw: dict) -> Screen:
    return Screen(
        raw["url"],
        raw["title"],
        tuple(Control(**{**c, "box": tuple(c["box"]), "point": tuple(c["point"]) if c["point"] else None}) for c in raw["controls"]),
        tuple(Text(**{**t, "box": tuple(t["box"])}) for t in raw["texts"]),
        tuple(Table(t["label"], tuple(t["header"]), tuple(tuple(r) for r in t["rows"])) for t in raw["tables"]),
        raw["dialog"],
        raw["busy"],
        tuple(Graphic(g["label"], tuple(g["box"])) for g in raw["graphics"]),
    )


# ------------------------------------------------------------------- actions


@tc.action(effect="external", idempotent=False)
@dataclass(frozen=True)
class Click:
    target: str
    x: int
    y: int


@tc.action(effect="external", idempotent=True)
@dataclass(frozen=True)
class TypeText:
    target: str
    text: str
    submit: bool = False
    replace: bool = True  # select the field's content first; only ever sent once focus is confirmed in an editable field


@tc.action(effect="external", idempotent=False)
@dataclass(frozen=True)
class PressKey:
    key: str  # a Playwright key name, e.g. "Backspace"


@tc.action(effect="read", idempotent=True)
@dataclass(frozen=True)
class Scroll:
    dy: int


FOCUS_JS = r"""
([x, y]) => {
  const a = document.activeElement;
  if (!a || a === document.body || a === document.documentElement) return false;
  const editable = a.isContentEditable || a.matches('textarea, input:not([type=checkbox]):not([type=radio]):not([type=button]):not([type=submit]):not([type=reset]):not([type=file]):not([type=image]):not([type=range]):not([type=color])');
  if (!editable || a.disabled || a.readOnly) return false;
  const r = a.getBoundingClientRect();
  if (x >= r.left - 2 && x <= r.right + 2 && y >= r.top - 2 && y <= r.bottom + 2) return true;
  const t = document.elementFromPoint(x, y);
  return !!t && (t.contains(a) || (!!a.labels && [...a.labels].some((l) => l.contains(t))));
}
"""

FIELD_TEXT_JS = r"""
({ name, box }) => {
  const editable = (e) => e && (e.isContentEditable || e.matches('textarea, input:not([type=checkbox]):not([type=radio]):not([type=button]):not([type=submit]):not([type=reset]):not([type=file]):not([type=image])'));
  const overlaps = (e) => { const r = e.getBoundingClientRect(); return r.right > box[0] && r.left < box[0] + box[2] && r.bottom > box[1] && r.top < box[1] + box[3]; };
  let el = name ? [...document.querySelectorAll('[aria-label]')].find((e) => e.getAttribute('aria-label') === name && editable(e)) : null;
  if (!el) el = [...document.querySelectorAll('input, textarea, [contenteditable]')].find((e) => editable(e) && overlaps(e));
  if (!el) return null;
  return el.isContentEditable ? el.innerText : el.value;
}
"""



@dataclass
class PageExecutor:
    """Executes UI actions on a Playwright page with real input events."""

    page: object

    def execute(self, act: object, *, key: str | None) -> tc.Receipt:
        page = self.page
        if isinstance(act, Click):
            page.mouse.click(act.x, act.y)
        elif isinstance(act, TypeText):
            if act.replace:
                page.keyboard.press("ControlOrMeta+A")
            page.keyboard.type(act.text)
            if act.submit:
                page.keyboard.press("Enter")
        elif isinstance(act, PressKey):
            page.keyboard.press(act.key)
        elif isinstance(act, Scroll):
            page.mouse.wheel(0, act.dy)
        else:
            return tc.Receipt(act, "rejected", error="unsupported action")
        return tc.Receipt(act, "applied", idempotency_key=key)


@dataclass
class Stats:
    actions: int = 0
    observations: int = 0
    browser_s: float = 0.0  # time inside Playwright calls (input dispatch + DOM reads + settling)
    started: float = field(default_factory=time.perf_counter)


class Browser:
    """What an agent program sees: observe, click, type, and caption. Nothing else."""

    def __init__(self, page: object, *, episode: str, on_step: Callable[[], None] | None = None) -> None:
        self.page = page
        self.executor = PageExecutor(page)
        self.episode = episode
        self.on_step = on_step or (lambda: None)
        self.stats = Stats()
        self._step = 0

    def observe(self, *, settle_ms: int = 2000) -> Screen:
        t0 = time.perf_counter()
        deadline = t0 + settle_ms / 1000
        raw = self.page.evaluate(PERCEIVE_JS)
        while raw["busy"] and time.perf_counter() < deadline:
            self.page.wait_for_timeout(15)
            raw = self.page.evaluate(PERCEIVE_JS)
        self.stats.browser_s += time.perf_counter() - t0
        self.stats.observations += 1
        return _screen(raw)

    def _do(self, act: object) -> tc.Receipt:
        self._step += 1
        t0 = time.perf_counter()
        receipt = tc.invoke(act, executor=self.executor, key=f"{self.episode}:{self._step}")
        self.stats.browser_s += time.perf_counter() - t0
        self.stats.actions += 1
        self.on_step()
        return receipt

    def click(self, control: Control, *, wait_ms: int = 3000) -> tc.Receipt:
        """Click where the control is actually hittable; if something covers it, wait (bounded) for it to clear."""
        deadline = time.perf_counter() + wait_ms / 1000
        while control.point is None:
            if time.perf_counter() > deadline:
                return tc.Receipt(Click(control.name or control.role, -1, -1), "rejected", error="control stayed covered")
            self.page.wait_for_timeout(100)
            same = [c for c in self.observe().controls if (c.role, c.name, c.section) == (control.role, control.name, control.section)]
            if not same:
                return tc.Receipt(Click(control.name or control.role, -1, -1), "rejected", error="control disappeared")
            control = same[0]
        x, y = control.point
        return self._do(Click(control.name or control.role, x, y))

    def fill(self, control: Control, text: str, *, submit: bool = False) -> tc.Receipt:
        """Verified typing: click, confirm the keyboard is going into the field, then select-all and type.

        If focus is not confirmed, the control is perceived again (its point may be stale after a
        re-render) and clicked once more, then at a different point in the box. If focus still
        cannot be confirmed, nothing is typed and the receipt is ``rejected`` with the reason, so
        the mind can escalate instead of waiting on keystrokes that went nowhere.
        """
        tried: list[tuple[int, int]] = []
        for step in range(3):
            if step == 0:
                clicked = self.click(control)
                if clicked.status == "rejected":
                    return tc.Receipt(TypeText(control.name, text, submit), "rejected", error=f"could not click the field: {clicked.error}")
                point = (clicked.action.x, clicked.action.y)
            else:
                if step == 1:  # the point may be stale: look again before deciding it cannot be typed into
                    fresh = self.same_control(control)
                    if fresh is None or fresh.point is None:
                        continue
                    control, point = fresh, fresh.point
                else:
                    point = self.other_point(control, tried[-1])
                if point is None or point in tried:
                    continue
                self._do(Click(control.name or control.role, *point))
            tried.append(point)
            if self.focused(control, point):
                return self._do(TypeText(control.name, text, submit))
        return tc.Receipt(TypeText(control.name, text, submit), "rejected",
                          error=f"keyboard focus not confirmed in {control.name or control.role!r} after clicking {', '.join(str(p) for p in tried)}")

    def same_control(self, control: Control) -> Control | None:
        """The same control as perceived now (role, name and section), with fresh geometry."""
        return next((c for c in self.observe().controls if (c.role, c.name, c.section) == (control.role, control.name, control.section)), None)

    def focused(self, control: Control, point: tuple[int, int]) -> bool:
        """Is the keyboard going into this field?

        Either the active element is an editable field at (or labelled by what is at) the
        clicked point, or a probe keystroke lands in the field's value: some apps route keys
        through a window-level handler, so an editable ``activeElement`` is not required. The
        probe character is removed again, wherever it went.
        """
        for _ in range(5):
            if self.page.evaluate(FOCUS_JS, list(point)):
                return True
            self.page.wait_for_timeout(40)
        before = self.field_text(control)
        self._do(TypeText(control.name, "x", False, replace=False))
        self.page.wait_for_timeout(60)
        after = self.field_text(control)
        self._do(PressKey("Backspace"))  # undo the probe wherever it landed
        self.page.wait_for_timeout(40)
        return after is not None and after != before

    def field_text(self, control: Control) -> str | None:
        """What the field currently holds, or None if it cannot be read (then focus stays unconfirmed)."""
        return self.page.evaluate(FIELD_TEXT_JS, {"name": control.name, "box": list(control.box)})

    @staticmethod
    def other_point(control: Control, tried: tuple[int, int]) -> tuple[int, int] | None:
        x, y, w, h = control.box
        options = [(round(x + w * fx), round(y + h / 2)) for fx in (0.15, 0.5, 0.85, 0.96)]
        best = max(options, key=lambda p: abs(p[0] - tried[0]) + abs(p[1] - tried[1]))
        return best if abs(best[0] - tried[0]) + abs(best[1] - tried[1]) >= 4 else None

    def scroll(self, dy: int) -> tc.Receipt:
        return self._do(Scroll(dy))

    def caption(self, text: str) -> None:
        self.page.evaluate("t => window.__tcCaption && window.__tcCaption(t)", text)

    def screenshot(self, box: tuple[int, int, int, int]) -> bytes:
        x, y, w, h = box
        return self.page.screenshot(clip={"x": x, "y": y, "width": w, "height": h}, type="png")


# ----------------------------------------------------------------- targeting

_WORD = re.compile(r"[a-z0-9]+")


def _norm(s: str) -> str:
    return " ".join(_WORD.findall(s.lower()))


def _trigrams(s: str) -> set[str]:
    s = f"  {_norm(s)} "
    return {s[i : i + 3] for i in range(len(s) - 2)}


def label_similarity(query: str, name: str) -> float:
    q, n = _norm(query), _norm(name)
    if not q or not n:
        return 0.0
    if q == n:
        return 1.0
    qt, nt = set(q.split()), set(n.split())
    if qt <= nt:
        return 0.9
    a, b = _trigrams(q), _trigrams(n)
    return 0.8 * (2 * len(a & b) / (len(a) + len(b)))


@dataclass
class LabelMatcher:
    """Ranks controls by label similarity to any of several synonyms. Rules, not a model."""

    name: str = "label-matcher"
    version: str = "1"
    op: str = "rank"
    traits: tc.Traits = RULES
    profile: tc.Profile = IN_PROCESS

    def accepts(self, request: tc.Request) -> bool:
        return request.op == "rank" and isinstance(request.subject, tuple) and all(isinstance(c, Control) for c in request.params["candidates"])

    def run(self, requests: Sequence[tc.Request]) -> list[tc.Output]:
        outs = []
        for r in requests:
            scored = [(c, tc.Score(max(label_similarity(q, f"{c.name} {c.group}".strip()) for q in r.subject), "similarity")) for c in r.params["candidates"]]
            outs.append(tc.Output(sorted(scored, key=lambda p: p[1].value, reverse=True)))
        return outs


def find(screen: Screen, *labels: str, roles: Iterable[str] = (), min_score: float = 0.55, margin: float = 0.08) -> Control | tc.Unknown:
    """The control whose label best matches; Unknown if nothing matches clearly."""
    roles = set(roles)
    candidates = [c for c in screen.controls if not c.disabled and (not roles or c.role in roles)]
    if not candidates:
        return tc.Unknown("no_candidate_controls", f"roles={sorted(roles)}")
    ranked = tc.rank(tuple(labels), candidates)
    if isinstance(ranked, tc.Unknown):
        return ranked
    (best, s1), rest = ranked[0], ranked[1:]
    if s1.value < min_score:
        return tc.Unknown("no_matching_control", f"best {best.name!r} scored {s1.value:.2f} for {labels}", tuple(ranked[:3]))
    if rest and s1.value - rest[0][1].value < margin and best.name != rest[0][0].name:
        return tc.Unknown("ambiguous_control", f"{best.name!r} vs {rest[0][0].name!r}", tuple(ranked[:3]))
    return best


# ---------------------------------------------------- verified, bounded submit



class PageOutcome(enum.Enum):
    confirmed = "confirmed"
    transient_failure = "transient_failure"  # the page says nothing was saved
    unconfirmed = "unconfirmed"  # the page says it could not confirm
    invalid_input = "invalid_input"
    no_feedback = "no_feedback"


@dataclass(frozen=True)
class SubmitAttempt:
    new_alerts: tuple[str, ...]  # alerts that appeared since the click (old toasts linger)
    success: str


@tc.implementation("classify", name="page-feedback-rules", version="1", accepts=lambda r: isinstance(r.subject, SubmitAttempt) and r.target is PageOutcome, profile=IN_PROCESS)
def classify_feedback(request: tc.Request) -> PageOutcome:
    s: SubmitAttempt = request.subject
    alerts = " ".join(s.new_alerts).lower()
    if s.success.lower() in alerts:
        return PageOutcome.confirmed
    if "couldn't confirm" in alerts or "could not confirm" in alerts:
        return PageOutcome.unconfirmed
    if any(w in alerts for w in ("503", "unavailable", "nothing was saved", "try again")):
        return PageOutcome.transient_failure
    if any(w in alerts for w in ("must be", "required", "invalid")):
        return PageOutcome.invalid_input
    return PageOutcome.no_feedback


@dataclass(frozen=True)
class SubmitResult:
    status: str  # "done" | "escalated"
    reason: str
    attempts: int


def submit(ui: Browser, button: Control, *, success: str, already_done: Callable[[Screen], bool], max_attempts: int = 3) -> SubmitResult:
    """Click a non-idempotent submit button until the page confirms, retrying only when that is safe."""
    for attempt in range(1, max_attempts + 1):
        current = ui.observe()  # re-perceive every attempt: overlays move, points go stale
        before = current.last_alert
        button = next((c for c in current.controls if (c.role, c.name, c.section) == (button.role, button.name, button.section)), button)
        ui.click(button)
        screen = ui.observe()
        new_alerts = tuple(screen.alerts(after=before))
        outcome = tc.classify(SubmitAttempt(new_alerts, success), PageOutcome)
        if outcome is PageOutcome.confirmed:
            return SubmitResult("done", "page confirmed", attempt)
        if outcome is PageOutcome.unconfirmed:
            if already_done(screen):  # look before repeating an uncertain effect
                return SubmitResult("done", "found in submitted list after unconfirmed reply", attempt)
            continue
        if outcome is PageOutcome.transient_failure:
            continue
        return SubmitResult("escalated", f"{outcome.value if isinstance(outcome, PageOutcome) else outcome.reason}: {' | '.join(new_alerts)}", attempt)
    return SubmitResult("escalated", "attempt limit reached", max_attempts)
