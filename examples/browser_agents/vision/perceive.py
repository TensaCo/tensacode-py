"""Pixels -> scene graph: a screenshot becomes the same ``Screen`` the DOM perceiver makes.

    words  (OCR)            ─┐
    icons  (UI detector)    ─┼─> phrases, rectangles, windows ─> controls + texts ─> Screen
    edges  (classical CV)   ─┘                                                     └> Fragment (scored)

Nothing here reads a DOM or an accessibility API, so it applies to any screen image.
Every inferred element carries an uncalibrated confidence and the rule that produced it;
the Fragment attaches those as ``tc.Score`` so beliefs built on vision stay inspectable.

What pixels cannot give (and this module does not pretend to): names of icon-only
controls with no visible label (unless an ``IconMemory`` has seen them), hidden state
(disabled, aria-current), text under other windows, and exact z-order of windows.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

import tensorcode as tc
from tensorcode.backends.builtin import IN_PROCESS
from tensorcode.cognition import Fragment

from ..browser import Control, Screen, Text
from .models import Word

Box = tuple[int, int, int, int]
SCREEN = tc.Ref("scope:screen")
# a shell prompt: user@host:path followed by $ (OCR often reads it as 5, S or §), or PowerShell's "PS C:\...>"
# a shell prompt: user@host:path then $ . OCR reads "~" as -, _, " or #, and "$" as 5, S or §.
# The last alternative is a bare "$" prompt (PS1="$ "), which is what the computerworld
# terminal draws: a lone "$" at the start of a line inside a window is a shell prompt. It is
# kept strict (only "$", not its OCR confusions) so a price or a quoted line elsewhere on a
# screen is not mistaken for a place to type.
PROMPT = re.compile(r'^[\w.-]+@[\w.-]+[:;]\s?[~\-_/"#](?:[^\s]*?|[^$%>§]{0,120}?)(?:[$#%>5S§](?=\s|$)|$)\s?|^PS [A-Z]:\\.*?>\s?|^\$(?=\s|$)\s?')


# ------------------------------------------------------------------ geometry


def center(b: Box) -> tuple[float, float]:
    return b[0] + b[2] / 2, b[1] + b[3] / 2


def inside(pt: tuple[float, float], b: Box, pad: float = 0) -> bool:
    return b[0] - pad <= pt[0] <= b[0] + b[2] + pad and b[1] - pad <= pt[1] <= b[1] + b[3] + pad


def iou(a: Box, b: Box) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union else 0.0


def union_box(boxes: Sequence[Box]) -> Box:
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes)
    y1 = max(b[1] + b[3] for b in boxes)
    return (x0, y0, x1 - x0, y1 - y0)


def contains(outer: Box, inner: Box, pad: int = 2) -> bool:
    return outer[0] - pad <= inner[0] and outer[1] - pad <= inner[1] and inner[0] + inner[2] <= outer[0] + outer[2] + pad and inner[1] + inner[3] <= outer[1] + outer[3] + pad


# ------------------------------------------------------------------ elements


@dataclass(frozen=True)
class Phrase:
    """Words on one baseline with no large gap: a label, a cell, a line of terminal output."""

    text: str
    box: Box
    conf: float
    words: tuple[Word, ...]


@dataclass(frozen=True)
class Window:
    title: str
    box: Box
    complete: bool  # all four edges seen (likely frontmost)
    conf: float


@dataclass(frozen=True)
class Inferred:
    control: Control
    conf: float
    rule: str


@dataclass
class PixelScene:
    size: tuple[int, int]  # width, height
    phrases: list[Phrase]
    rects: list[Box]
    icons: list[tuple[Box, float]]
    windows: list[Window]
    controls: list[Inferred]
    texts: list[tuple[Text, float]]
    timings_ms: dict[str, float] = field(default_factory=dict)

    def screen(self, url: str = "pixels://screen", title: str = "") -> Screen:
        return Screen(url, title, tuple(i.control for i in self.controls), tuple(t for t, _ in self.texts), (), "", False, ())


# ------------------------------------------------------------------ phrases


def phrases_from_words(words: Sequence[Word], gap_ratio: float = 0.9) -> list[Phrase]:
    """Group words by the recognizer's lines, then split each line at wide gaps (columns, separate labels)."""
    by_line: dict[int, list[Word]] = {}
    loose: list[Word] = []
    for w in words:
        (by_line.setdefault(w.line, []) if w.line >= 0 else loose).append(w)
    groups = list(by_line.values())
    # words without a line id: cluster by vertical center
    for w in sorted(loose, key=lambda w: (center(w.box)[1], w.box[0])):
        for g in groups:
            if abs(center(g[0].box)[1] - center(w.box)[1]) < 0.5 * max(g[0].box[3], w.box[3]):
                g.append(w)
                break
        else:
            groups.append([w])
    out: list[Phrase] = []
    for g in groups:
        g = sorted(g, key=lambda w: w.box[0])
        run = [g[0]]
        for w in g[1:]:
            prev = run[-1]
            h = max(1, min(prev.box[3], w.box[3]))
            gap = w.box[0] - (prev.box[0] + prev.box[2])
            same_row = abs(center(prev.box)[1] - center(w.box)[1]) < 0.6 * h
            if gap > gap_ratio * h or not same_row:
                out.append(_phrase(run))
                run = [w]
            else:
                run.append(w)
        out.append(_phrase(run))
    return sorted(out, key=lambda p: (p.box[1], p.box[0]))


def _phrase(ws: list[Word]) -> Phrase:
    return Phrase(" ".join(w.text for w in ws), union_box([w.box for w in ws]), min(w.conf for w in ws), tuple(ws))


# ------------------------------------------------------------------ rectangles


def find_rects(rgb: np.ndarray) -> list[Box]:
    """Axis-aligned rectangles from edges: fields, buttons, panels, windows (rounded corners allowed)."""
    import cv2

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 12, 40)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8))
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    H, W = gray.shape
    rects: list[Box] = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w < 10 or h < 10 or (w > W - 4 and h > H - 4):
            continue
        area = cv2.contourArea(c)
        if area < 0.72 * w * h:
            continue
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) > 10:
            continue
        rects.append((x, y, w, h))
    rects.sort(key=lambda b: b[2] * b[3])
    kept: list[Box] = []
    for r in rects:
        if not any(iou(r, k) > 0.85 for k in kept):
            kept.append(r)
    return kept


def find_windows(rgb: np.ndarray, phrases: Sequence[Phrase]) -> list[Window]:
    """Windows: a long top edge meeting a long vertical side edge, with a title-like phrase just below the top."""
    import cv2

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.int16)
    H, W = gray.shape
    dy = np.zeros((H, W), np.uint8)
    dx = np.zeros((H, W), np.uint8)
    dy[1:] = np.abs(np.diff(gray, axis=0)) > 14
    dx[:, 1:] = np.abs(np.diff(gray, axis=1)) > 14
    horizontal = cv2.morphologyEx(dy, cv2.MORPH_OPEN, np.ones((1, 120), np.uint8))
    vertical = cv2.morphologyEx(dx, cv2.MORPH_OPEN, np.ones((90, 1), np.uint8))
    horizontal = cv2.dilate(horizontal, np.ones((3, 1), np.uint8))
    vertical = cv2.dilate(vertical, np.ones((1, 3), np.uint8))
    _, _, hstats, _ = cv2.connectedComponentsWithStats(horizontal, connectivity=8)
    _, _, vstats, _ = cv2.connectedComponentsWithStats(vertical, connectivity=8)
    hs = [tuple(int(v) for v in st[:4]) for st in hstats[1:] if st[2] >= 200]
    vs = [tuple(int(v) for v in st[:4]) for st in vstats[1:] if st[3] >= 120]
    candidates: list[tuple[Box, bool]] = []
    for hx, hy, hw, hh in hs:
        top = hy + hh // 2
        left = [v for v in vs if abs(v[1] - top) <= 8 and abs(v[0] + v[2] // 2 - hx) <= 8]
        right = [v for v in vs if abs(v[1] - top) <= 8 and abs(v[0] + v[2] // 2 - (hx + hw)) <= 8]
        if not (left or right):
            continue
        bottom = max(v[1] + v[3] for v in left + right)
        if bottom - top < 140 or (hw > W - 8 and bottom - top > H - 40):
            continue
        box = (hx, top, hw, bottom - top)
        closed = any(abs(b[1] + b[3] // 2 - bottom) <= 8 and b[0] <= hx + 8 and b[0] + b[2] >= hx + hw - 8 for b in hs)
        candidates.append((box, bool(left and right and closed)))
    windows: list[Window] = []
    for box, complete in sorted(candidates, key=lambda c: (not c[1], c[0][2] * c[0][3])):
        if any(iou(box, w.box) > 0.7 for w in windows):
            continue
        if any(other != box and contains(other, box, pad=4) and box[1] > other[1] + 20 for other, _ in candidates):
            continue  # a pane inside a window (a list, a message view, a composer), not a window
        strip = [p for p in phrases if box[1] - 4 <= p.box[1] and p.box[1] + p.box[3] <= box[1] + 52 and inside(center(p.box), box) and len(p.text) >= 2]
        above = [p for p in phrases if box[1] - 48 <= p.box[1] and p.box[1] + p.box[3] <= box[1] + 2 and box[0] <= center(p.box)[0] <= box[0] + box[2] and len(p.text) >= 2]
        centered = lambda ps: [p for p in ps if abs(center(p.box)[0] - (box[0] + box[2] / 2)) < 0.15 * box[2]]  # noqa: E731
        if centered(above) and not centered(strip) and box[2] < 0.9 * W:  # a screen-wide frame's "title above" is the system bar (a clock)
            # the edge found was the title bar's lower border (title bar blends into the desktop): the title sits above it
            top = min(p.box[1] for p in centered(above)) - 10
            box = (box[0], top, box[2], box[1] + box[3] - top)
            strip = centered(above)
        if not strip:
            continue
        cx = box[0] + box[2] / 2
        near_center = [p for p in strip if abs(center(p.box)[0] - cx) < 0.2 * box[2]]
        title = min(near_center, key=lambda p: abs(center(p.box)[0] - cx)) if near_center else min(strip, key=lambda p: p.box[0])
        windows.append(Window(title.text, box, complete, 0.6 if complete else 0.4))
    return windows


def split_at_windows(phrases: Sequence[Phrase], windows: Sequence[Window]) -> list[Phrase]:
    """A line of words cannot run across a window edge: split phrases whose words belong to different windows."""
    out: list[Phrase] = []
    for p in phrases:
        if len(p.words) < 2:
            out.append(p)
            continue
        run = [p.words[0]]
        for w in p.words[1:]:
            a, b = window_of(center(run[-1].box), windows), window_of(center(w.box), windows)
            if a is not b:
                out.append(_phrase(run))
                run = [w]
            else:
                run.append(w)
        out.append(_phrase(run))
    return sorted(out, key=lambda p: (p.box[1], p.box[0]))


def window_of(pt: tuple[float, float], windows: Sequence[Window]) -> Window | None:
    hits = [w for w in windows if inside(pt, w.box)]
    if not hits:
        return None
    return min(hits, key=lambda w: (not w.complete, w.box[2] * w.box[3]))


# ------------------------------------------------------------------ roles


def _contrast(rgb: np.ndarray, p: Phrase) -> float:
    """Luma difference between the phrase's darkest/lightest pixels: placeholder text is low contrast."""
    x, y, w, h = p.box
    patch = rgb[max(0, y):y + h, max(0, x):x + w].astype(np.float32)
    if patch.size == 0:
        return 0.0
    luma = patch @ np.array([0.299, 0.587, 0.114], np.float32)
    return float(np.percentile(luma, 95) - np.percentile(luma, 5))


def _checkbox_state(rgb: np.ndarray, r: Box) -> bool | None:
    """A checkbox is a square outline with a plain interior (unchecked) or a mostly filled interior (checked). Icons are neither."""
    x, y, w, h = r
    H, W = rgb.shape[:2]
    img = rgb.astype(np.int16)
    patch = img[y:y + h, x:x + w]
    if patch.shape[0] < 8 or patch.shape[1] < 8:
        return None
    around = np.concatenate([img[max(0, y - 3), max(0, x - 3):x + w + 3], img[min(H - 1, y + h + 2), max(0, x - 3):x + w + 3]])
    background = np.median(around, axis=0)
    inner = patch[3:-3, 3:-3].reshape(-1, 3)
    inner_mode = np.median(inner, axis=0)
    inner_spread = float(np.abs(inner - inner_mode).sum(-1).mean())
    # a circle's top edge is short; a (rounded) square's runs most of the width
    top_band = np.abs(img[y:y + 3, x:x + w] - background).sum(-1).max(0) > 60
    if top_band.mean() < 0.6:
        return None  # a dot or a circle: not a square box
    if inner_spread < 14 and float(np.abs(inner_mode - background).sum()) < 30 and _band_contrast(img, r, inner_mode) > 40:
        return False  # empty square: an outline around page-colored interior
    filled = float((np.abs(inner - background).sum(-1) > 90).mean())
    if 0.4 < filled < 0.95 and 15 <= inner_spread < 120:
        return True  # filled square with a mark in it
    return None


def _band_contrast(img: np.ndarray, r: Box, ref: np.ndarray, band: int = 3) -> float:
    """Weakest side of a box outline: for each side, the most distinct line within ``band`` px of the edge."""
    x, y, w, h = r
    H, W = img.shape[:2]
    sides = []
    for k in range(4):
        best = 0.0
        for d in range(0, band):
            if k == 0:
                line = img[min(H - 1, max(0, y + d)), x + 3:x + w - 3]
            elif k == 1:
                line = img[min(H - 1, max(0, y + h - 1 - d)), x + 3:x + w - 3]
            elif k == 2:
                line = img[y + 3:y + h - 3, min(W - 1, max(0, x + d))]
            else:
                line = img[y + 3:y + h - 3, min(W - 1, max(0, x + w - 1 - d))]
            if line.size:
                best = max(best, float(np.abs(np.median(line, axis=0) - ref).sum()))
        sides.append(best)
    return min(sides)


def _outline(rgb: np.ndarray, r: Box) -> tuple[bool, bool]:
    """(has a border line distinct from its interior on all four sides, interior differs from the surroundings)."""
    x, y, w, h = r
    H, W = rgb.shape[:2]
    if w < 8 or h < 8:
        return False, False
    img = rgb.astype(np.int16)
    interior = img[y + 4:y + h - 4, x + 4:x + w - 4].reshape(-1, 3)
    outside = np.concatenate([img[max(0, y - 4), x:x + w], img[min(H - 1, y + h + 3), x:x + w]])
    if interior.size == 0 or outside.size == 0:
        return False, False
    mode = np.median(interior, axis=0)
    border = _band_contrast(img, r, mode) > 30
    filled = float(np.abs(mode - np.median(outside, axis=0)).sum()) > 30
    return border, filled


def _label_for(field_box: Box, phrases: Sequence[Phrase], used: set[int], rects: Sequence[Box] = ()) -> str:
    fx, fy, fw, fh = field_box
    best, score = "", 1e9
    for i, p in enumerate(phrases):
        if i in used:
            continue
        if any(inside(center(p.box), r) and 0.5 < (r[2] * r[3]) / max(1, fw * fh) < 2 for r in rects if r != field_box):
            continue  # text inside a sibling box of similar size is that box's content, not this field's label
        px, py, pw, ph = p.box
        if len(p.text) > 60:
            continue
        # above, left-aligned-ish
        if 0 <= fy - (py + ph) <= 30 and abs(px - fx) <= 40:
            s = fy - (py + ph) + abs(px - fx) * 0.3
        # left, same row
        elif abs(center(p.box)[1] - (fy + fh / 2)) < fh / 2 and 0 <= fx - (px + pw) <= 140:
            s = (fx - (px + pw)) * 0.5
        else:
            continue
        if s < score:
            best, score = p.text, s
    return best


def infer_controls(rgb: np.ndarray, phrases: list[Phrase], rects: list[Box], icons: list[tuple[Box, float]], windows: list[Window],
                   icon_namer: Callable[[np.ndarray, Box], tuple[str, float]] | None = None, text_candidates: bool = True) -> tuple[list[Inferred], list[tuple[Text, float]]]:
    H, W = rgb.shape[:2]
    out: list[Inferred] = []
    used_phrases: set[int] = set()
    taken: list[Box] = []

    def section(b: Box) -> str:
        w = window_of(center(b), windows)
        return w.title if w else ""

    def add(role: str, name: str, box: Box, conf: float, rule: str, *, value: str = "", hint: str = "", checked: bool | None = None) -> None:
        x, y, w, h = box
        pt = (int(min(W - 1, max(0, x + w / 2))), int(min(H - 1, max(0, y + h / 2))))
        c = Control(role, name, value, checked, False, hint, "", section(box), "", box, pt, False, "")
        out.append(Inferred(c, conf, rule))
        taken.append(box)

    phrase_centers = [center(p.box) for p in phrases]

    # 1) checkboxes: small squares immediately left of a phrase
    for r in rects:
        x, y, w, h = r
        if not (12 <= w <= 24 and 12 <= h <= 24 and abs(w - h) <= 3):
            continue
        right = [(i, p) for i, p in enumerate(phrases) if i not in used_phrases and 0 <= p.box[0] - (x + w) <= 16 and abs(center(p.box)[1] - (y + h / 2)) < h]
        if not right:
            continue
        state = _checkbox_state(rgb, r)
        if state is None:
            continue
        i, p = min(right, key=lambda ip: ip[1].box[0])
        used_phrases.add(i)
        add("checkbox", p.text, r, 0.6, "square-box-left-of-phrase", checked=state)

    # 2) field-like and button-like rectangles
    for r in rects:
        x, y, w, h = r
        if not (16 <= h <= 64 and w >= 18 and w <= 0.8 * W):
            continue
        if any(contains(t, r) and t[2] * t[3] < 4 * w * h for t in taken):
            continue
        inner = [i for i, c in enumerate(phrase_centers) if i not in used_phrases and inside(c, r) and phrases[i].box[3] < h + 4]
        border, filled = _outline(rgb, r)
        holds_control = any(contains(r, t, pad=0) and t[2] * t[3] < 0.6 * w * h for t in taken)
        if holds_control:
            free = [i for i, c in enumerate(phrase_centers) if i not in used_phrases and inside(c, r)]
            if free and h <= 48:  # a list row: its own label plus an inner control (a status pill, a toggle)
                everything = sorted((p for p in phrases if inside(center(p.box), r)), key=lambda p: p.box[0])
                used_phrases.update(free)
                add("button", " ".join(p.text for p in everything), r, 0.5, "row-holding-a-control")
            continue  # otherwise a toolbar or panel: its controls speak for themselves
        if inner:
            ps = sorted((phrases[i] for i in inner), key=lambda p: p.box[0])
            tb = union_box([p.box for p in ps])
            text = " ".join(p.text for p in ps)
            gaps = [b.box[0] - (a.box[0] + a.box[2]) for a, b in zip(ps, ps[1:])]
            one_label = not gaps or max(gaps) < 2.5 * h
            centered = abs(center(tb)[0] - (x + w / 2)) < 0.18 * w
            left = tb[0] - x < 0.3 * w
            if not one_label:
                continue  # a bar holding several labels (a toolbar, a footer): look at them one by one
            if centered and tb[2] >= 0.3 * w and (border or filled):
                used_phrases.update(inner)
                add("button", text, r, 0.7, "centered-text-in-outlined-rect")
            elif left and w >= 2.5 * h and border:
                used_phrases.update(inner)
                faint = _contrast(rgb, ps[0]) < 110
                label = _label_for(r, phrases, used_phrases, rects)
                add("textbox", label, r, 0.55, "left-text-in-outlined-field", value="" if faint else text, hint=text if faint else "")
            elif left and filled and not border:
                used_phrases.update(inner)
                add("button", text, r, 0.55, "text-in-filled-row")
        elif w >= 3 * h and h <= 48 and border:
            label = _label_for(r, phrases, used_phrases, rects)
            if label or w >= 120:
                add("textbox", label, r, 0.4, "empty-outlined-field")

    # 3) detected interactable elements
    for box, conf in sorted(icons, key=lambda ic: -ic[1]):
        if box[2] > 420 or box[3] > 90 or any(iou(box, t) > 0.5 or contains(t, box) and t[2] * t[3] < 3 * box[2] * box[3] for t in taken):
            continue
        if conf < 0.12:  # weak detections survive only as icons someone has named before
            name, sim = icon_namer(rgb, box) if icon_namer else ("", 0.0)
            if name and not any(inside(c, box) for c in phrase_centers):
                add("button", name, box, 0.5 * sim, "icon-memory-weak-detection")
            continue
        inner = [i for i, c in enumerate(phrase_centers) if i not in used_phrases and inside(c, box, pad=2)]
        if inner:
            used_phrases.update(inner)
            covered = sum(phrases[i].box[2] for i in inner) >= 0.4 * box[2]
            add("button", " ".join(phrases[i].text for i in sorted(inner, key=lambda i: phrases[i].box[0])), box, conf if covered else conf * 0.8, "detector-with-text" if covered else "detector-row-with-label")
            continue
        # a label beside the icon (sidebars, menus) or below it (desktop and app-grid icons)
        bx, by, bw, bh = box
        beside = [i for i, p in enumerate(phrases) if i not in used_phrases and 0 <= p.box[0] - (bx + bw) <= 18 and abs(center(p.box)[1] - (by + bh / 2)) < max(bh, p.box[3]) * 0.6 and len(p.text) <= 40]
        below = [] if min(bw, bh) < 28 or not 0.6 <= bw / bh <= 1.6 else [i for i, p in enumerate(phrases) if i not in used_phrases and 0 <= p.box[1] - (by + bh) <= 14 and abs(center(p.box)[0] - (bx + bw / 2)) < max(bw, p.box[2]) * 0.6 and len(p.text) <= 40]
        if beside:
            i = min(beside, key=lambda i: phrases[i].box[0])
            used_phrases.add(i)
            add("button", phrases[i].text, union_box([box, phrases[i].box]), conf * 0.9, "icon-with-label-beside")
        elif below:
            i = min(below, key=lambda i: phrases[i].box[1])
            used_phrases.add(i)
            add("button", phrases[i].text, union_box([box, phrases[i].box]), conf * 0.9, "icon-with-label-below")
        else:
            name, sim = icon_namer(rgb, box) if icon_namer else ("", 0.0)
            if name:
                add("button", name, box, conf * (0.5 + 0.5 * sim), "icon-memory")
            elif conf >= 0.3:
                add("button", "", box, conf * 0.5, "icon-unnamed")

    # 4) a command prompt is where typing goes
    for i, p in enumerate(phrases):
        if PROMPT.match(p.text):
            win = window_of(center(p.box), windows)
            last = win is None or not any(PROMPT.match(q.text) and q.box[1] > p.box[1] and inside(center(q.box), win.box) for q in phrases)
            if last:
                # typing goes after the prompt: the input starts where the prompt text ends (estimated per character)
                m = PROMPT.match(p.text)
                x0 = p.box[0] + round(p.box[2] * len(m.group(0).rstrip()) / max(1, len(p.text))) + 6
                x1 = (win.box[0] + win.box[2] - 8) if win else min(W, p.box[0] + 600)
                if x1 - x0 >= 20:
                    add("textbox", "", (x0, p.box[1] - 3, x1 - x0, p.box[3] + 6), 0.5, "command-prompt-line", hint="command prompt", value=PROMPT.sub("", p.text))

    # 5) short phrases may be clickable text (menus, tabs, list rows): low-confidence candidates
    if text_candidates:
        consoles = [w for w in windows if any(PROMPT.match(q.text) and inside(center(q.box), w.box) for q in phrases)]
        for i, p in enumerate(phrases):
            if i in used_phrases or len(p.text) > 32 or len(p.text.split()) > 5 or len(p.text) < 2:
                continue
            if any(inside(center(p.box), w.box) and p.box[1] > w.box[1] + 50 for w in consoles):
                continue
            if any(inside(center(p.box), t) for t in taken):
                continue
            add("button", p.text, (p.box[0] - 4, p.box[1] - 3, p.box[2] + 8, p.box[3] + 6), 0.25 * p.conf, "short-text-candidate")

    texts = [(Text("text", p.text, section(p.box), p.box, 0), p.conf) for p in phrases]
    return out, texts


# ------------------------------------------------------------------ pipeline


def perceive(rgb: np.ndarray, *, ocr: Callable[[np.ndarray], list[Word]], detector: Callable[[np.ndarray], list[tuple[Box, float]]] | None = None,
             icon_namer: Callable[[np.ndarray, Box], tuple[str, float]] | None = None, text_candidates: bool = True) -> PixelScene:
    import time

    t: dict[str, float] = {}
    t0 = time.perf_counter()
    words = ocr(rgb)
    t["ocr"] = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter()
    icons = detector(rgb) if detector else []
    t["detector"] = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter()
    phrases = phrases_from_words(words)
    rects = find_rects(rgb)
    windows = find_windows(rgb, phrases)
    phrases = split_at_windows(phrases, windows)
    t["cv"] = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter()
    controls, texts = infer_controls(rgb, phrases, rects, icons, windows, icon_namer, text_candidates)
    t["roles"] = (time.perf_counter() - t0) * 1e3
    return PixelScene((rgb.shape[1], rgb.shape[0]), phrases, rects, icons, windows, controls, texts, t)


# ------------------------------------------------------------------ as tensorcode claims


@dataclass(frozen=True)
class Screenshot:
    rgb: np.ndarray = field(repr=False)
    frame: int = 0


def scene_fragment(scene: PixelScene, frame: int = 0) -> Fragment:
    """The DOM scene graph's vocabulary (is_a, in, label, hint, value, checked, reads) plus windows, with scores."""
    claims: list[tuple[tc.Claim, str | None]] = []
    entities: list[tuple[tc.Ref, object]] = []
    confidence: dict[str, tc.Score] = {}
    seen: Counter[str] = Counter()

    def say(subject: tc.Ref, predicate: str, obj: object, where: str, conf: float, basis: str) -> None:
        claim = tc.Claim(subject, predicate, obj, scope=SCREEN)
        claims.append((claim, where))
        confidence[claim.id] = tc.Score(round(conf, 3), "uncalibrated", basis)

    for n, w in enumerate(scene.windows):
        ref = tc.Ref(f"window:{w.title}#{n}")
        entities.append((ref, w))
        say(ref, "is_a", "window", f"box{w.box}", w.conf, "frame-edges+title-strip")
        say(ref, "title", w.title, f"box{w.box}", w.conf, "frame-edges+title-strip")
    for inf in scene.controls:
        c = inf.control
        key = f"{c.section}/{c.role}/{c.name}"
        seen[key] += 1
        ref = tc.Ref(f"ui:{key}#{seen[key]}")
        entities.append((ref, c))
        where = f"box{c.box}"
        say(ref, "is_a", c.role, where, inf.conf, inf.rule)
        say(ref, "in", c.section, where, inf.conf, inf.rule)
        if c.name:
            say(ref, "label", c.name, where, inf.conf, inf.rule)
        if c.hint:
            say(ref, "hint", c.hint, where, inf.conf, inf.rule)
        if c.role == "textbox":
            say(ref, "value", c.value, where, inf.conf, inf.rule)
        if c.checked is not None:
            say(ref, "checked", c.checked, where, inf.conf, inf.rule)
    sections: Counter[str] = Counter()
    for t, conf in scene.texts:
        sections[t.section] += 1
        ref = tc.Ref(f"text:{t.section}#{sections[t.section]}")
        entities.append((ref, t))
        say(ref, "reads", t.text, f"box{t.box}", conf, "ocr")
    return Fragment(tc.Ref(f"obs:pixels-{frame}"), tuple(claims), tuple(entities), snapshot_of=SCREEN, method="pixel-scene-graph@1", confidence=confidence)


def pixel_scene_graph(ocr: Callable, detector: Callable | None = None, icon_namer: Callable | None = None) -> tc.FunctionImplementation:
    """A ``parse`` implementation: Screenshot -> Fragment, for runtimes that perceive by pixels."""

    @tc.implementation("parse", name="pixel-scene-graph", version="1", accepts=lambda r: isinstance(r.subject, Screenshot) and r.target is Fragment, profile=IN_PROCESS)
    def parse(request: tc.Request) -> Fragment:
        shot: Screenshot = request.subject
        return scene_fragment(perceive(shot.rgb, ocr=ocr, detector=detector, icon_namer=icon_namer), shot.frame)

    return parse
