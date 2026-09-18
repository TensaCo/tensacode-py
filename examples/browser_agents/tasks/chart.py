"""Answer a question from a chart that exists only as pixels, as a mind.

Vision: the canvas is screenshotted and segmented (numpy) into bars: no OCR, no model.
Fusion: rules calibrate the y axis from the DOM tick labels and name each bar by the
nearest x label. Deliberation fills the answer form and submits once, verified.
"""

from __future__ import annotations

import io
import re

import numpy as np
from PIL import Image

import tensorcode as tc
from tensorcode.cognition import Fragment, Rule

from ..browser import Browser, PageOutcome, SubmitAttempt
from ..mind import BY_PRIORITY, Enter, Escalate, Finish, MindSpec, Press, Wait, controls, knowledge, objects, one

V = tc.Var
CHART, ANSWER, VISION = "Q3 revenue chart", tc.Ref("agent:answer"), tc.Ref("scope:vision:chart")


# ------------------------------------------------------------------ vision


def look_at_chart(ui: Browser, mind: tc.Store) -> Fragment | None:
    graphics = [r.claim.subject for r in mind.claims(predicate="is_a", object="graphic")]
    if not graphics or mind.claims(predicate="top_y"):
        return None
    x0, y0, w, h = mind.get(graphics[0]).box
    rgb = np.asarray(Image.open(io.BytesIO(ui.screenshot((x0, y0, w, h)))).convert("RGB")).astype(int)
    saturated = (rgb.max(2) - rgb.min(2)) > 60
    colors, counts = np.unique(rgb[saturated].reshape(-1, 3), axis=0, return_counts=True)
    if not len(colors):
        return None
    bar = colors[counts.argmax()]  # the dominant saturated color is the series
    near = np.abs(rgb - bar).sum(2) < 40
    columns = near.sum(0) >= 6  # thin dashed reference lines never fill a column
    claims = []
    edges = np.flatnonzero(np.diff(np.r_[0, columns.astype(int), 0]))
    for k, (a, b) in enumerate(zip(edges[::2], edges[1::2] - 1)):
        rows = np.flatnonzero(near[:, (a + b) // 2])
        ref = tc.Ref(f"bar:{k}")
        where = f"pixels x{a}-{b}"
        claims += [
            (tc.Claim(ref, "spans_x", (int(x0 + a), int(x0 + b)), scope=VISION), where),
            (tc.Claim(ref, "top_y", float(y0 + rows.min()), scope=VISION), where),
            (tc.Claim(ref, "bottom_y", float(y0 + rows.max() + 1), scope=VISION), where),
        ]
    return Fragment(tc.Ref("obs:canvas-pixels"), tuple(claims), snapshot_of=VISION, method="pixel-segmentation@1")


# ------------------------------------------------------ spontaneous thoughts


def _center(mind: tc.Store, ref: tc.Ref) -> tuple[float, float]:
    x, y, w, h = mind.get(ref).box
    return x + w / 2, y + h / 2


def _labels(mind: tc.Store) -> tuple[list[tuple[float, float]], list[tuple[str, float]]]:
    ticks, names = [], []
    for r in mind.claims(predicate="reads"):
        if not r.claim.subject.id.startswith(f"text:{CHART}#"):
            continue
        text = r.claim.object
        cx, cy = _center(mind, r.claim.subject)
        if m := re.fullmatch(r"(\d+)(k?)", text):
            ticks.append((cy, float(m[1])))
        elif re.fullmatch(r"[A-Z][a-z]+", text):
            names.append((text, cx))
    return sorted(set(ticks)), sorted(set(names))


def _bar_value(b, mind):
    ticks, names = _labels(mind)
    if len(ticks) < 2 or not names:
        return
    ys, vs = np.array([t[0] for t in ticks]), np.array([t[1] for t in ticks])
    slope, intercept = np.polyfit(ys, vs, 1)  # value per pixel, from the axis labels
    x0, x1 = b["span"]
    name = min(names, key=lambda n: abs(n[1] - (x0 + x1) / 2))[0]
    value = float(slope * b["top"] + intercept)
    yield tc.Claim(tc.Ref(f"region:{name}"), "q3_revenue", round(value, 1)), tc.Score(round(abs(slope), 3), "uncalibrated")


def _feedback(b, mind):
    outcome = tc.classify(SubmitAttempt((b["text"],), "Answer recorded"), PageOutcome)
    if isinstance(outcome, PageOutcome) and outcome is not PageOutcome.no_feedback and objects(mind, ANSWER, "attempt"):
        yield knowledge([(tc.Claim(ANSWER, "feedback", (b["a"].id, outcome.value)), None)], b["a"].id, "page-feedback-rules@1")


RULES = [
    Rule("bar_value_from_axis", ((V("bar"), "top_y", V("top")), (V("bar"), "spans_x", V("span"))), _bar_value),
    Rule("announcement_feedback", ((V("a"), "announces", V("text")),), _feedback),
]


# ------------------------------------------------------------ deliberation


def intentions(mind: tc.Store) -> list[object]:
    feedback = [v for _, v in objects(mind, ANSWER, "feedback")]
    if "confirmed" in feedback:
        return [Finish("answer recorded", priority=100)]
    revenue = {r.claim.subject.id[7:]: r.claim.object for r in mind.claims(predicate="q3_revenue")}
    _, names = _labels(mind)
    if not names or len(revenue) < len(names):
        return [Wait(20, "looking at the chart", priority=1)]
    ranked = sorted(revenue, key=revenue.get, reverse=True)
    top = ranked[0]
    per_pixel = max((r.evidence[0].confidence.value for r in mind.claims(predicate="q3_revenue") if r.evidence and r.evidence[0].confidence), default=0.0)
    if len(ranked) > 1 and revenue[top] - revenue[ranked[1]] < per_pixel / 2:
        # bar tops are whole pixels: equal pixel heights carry no information about which value is larger
        return [Escalate(f"too close to call from pixels: {top} and {ranked[1]} both read {revenue[top]}k (one pixel ≈ {per_pixel:.2f}k)")]
    diff = str(round(revenue[top] - min(revenue.values())))
    todo: list[object] = []
    combo = controls(mind, role="combobox", section="Answer")
    if combo and one(mind, combo[0], "shows") != top:
        options = controls(mind, role="option", label=top)
        todo.append(Press(options[0], f"top region := {top} ({revenue[top]}k)", priority=50) if options else Press(combo[0], "open regions", priority=49))
    for box in controls(mind, role="textbox", label="Difference ($k)"):
        if one(mind, box, "value") != diff:
            todo.append(Enter(box, diff, f"difference := {diff}k ({revenue[top]} − {min(revenue.values())})", priority=48))
    if todo:
        return todo
    attempts = len(objects(mind, ANSWER, "attempt"))
    if attempts > len(feedback):
        return [Wait(25, "waiting for confirmation", priority=5)]
    if attempts >= MAX_SUBMITS or attempts - feedback.count("transient_failure") >= MAX_UNSAFE_SUBMITS:
        return [Escalate(f"answer not confirmed after {attempts} submits ({', '.join(feedback)})")]
    return [Press(b, f"submit answer (attempt {attempts + 1})", priority=40, records=(tc.Claim(ANSWER, "attempt", attempts + 1),)) for b in controls(mind, role="button", label="Submit answer")]


MAX_UNSAFE_SUBMITS, MAX_SUBMITS = 3, 6  # attempts that may have recorded an answer / all attempts


def _safe(i: object, mind: tc.Store) -> bool:
    """A 503 that says nothing was saved is safe to retry; anything else counts toward the cap."""
    if not (isinstance(i, Press) and i.records):
        return True
    transient = [v for _, v in objects(mind, ANSWER, "feedback")].count("transient_failure")
    return i.records[0].object <= MAX_SUBMITS and (i.records[0].object - 1) - transient < MAX_UNSAFE_SUBMITS


SPEC = MindSpec("chart", RULES, intentions, BY_PRIORITY, constraints=(tc.Constraint("at_most_3_effectful_submits", _safe),), perceivers=(look_at_chart,), max_cycles=80)
BINDINGS: list = []
