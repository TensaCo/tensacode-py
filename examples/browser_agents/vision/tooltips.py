"""Learn icon names the way a person does: hover it, read the tooltip, remember the picture.

Pixels cannot say that a terminal glyph in a dock means "Terminal". A tooltip can. This
walks the icon-only controls a vision provider found, hovers each one, waits for a tooltip
to appear, reads it with OCR, and stores the name against the icon's thumbnail in an
``IconMemory``. Nothing is labelled by hand and no accessibility tree is consulted, so the
same routine works on any screen where hovering shows a label.

    learned, report = learn_from_tooltips(page, provider_scene, ocr, memory)

A tooltip is detected as *new text that appeared near the icon* between the frame before
the hover and the frame after it, which is what a tooltip is, whatever it is drawn with.
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass

import numpy as np

from .icon_memory import IconMemory
from .perceive import Box, PixelScene, inside


@dataclass
class TooltipLearning:
    hovered: int = 0
    tooltips_seen: int = 0
    names_learned: int = 0
    names: tuple[str, ...] = ()
    seconds: float = 0.0


def _grab(page) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(io.BytesIO(page.screenshot(type="png"))).convert("RGB"))


def _changed_boxes(before: np.ndarray, after: np.ndarray, min_pixels: int = 60) -> list[Box]:
    """Rectangles that appeared between two frames (a tooltip is one of them)."""
    import cv2

    if before.shape != after.shape:
        return []
    diff = (np.abs(after.astype(np.int16) - before).sum(-1) > 40).astype(np.uint8)
    diff = cv2.morphologyEx(diff, cv2.MORPH_CLOSE, np.ones((5, 9), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(diff, connectivity=8)
    out = []
    for st in stats[1:]:
        x, y, w, h, area = int(st[0]), int(st[1]), int(st[2]), int(st[3]), int(st[4])
        if area >= min_pixels and 8 <= h <= 90 and w >= 14:
            out.append((x, y, w, h))
    return out


def read_tooltip(page, icon: Box, ocr, *, settle_ms: int = 450, radius: int = 240) -> tuple[str, Box | None]:
    """Hover the icon and return the text that appeared next to it, if any."""
    x, y, w, h = icon
    before = _grab(page)
    page.mouse.move(x + w / 2, y + h / 2)
    page.wait_for_timeout(settle_ms)
    after = _grab(page)
    candidates = []
    for box in _changed_boxes(before, after):
        bx, by, bw, bh = box
        near = abs(bx - x) < radius and abs(by - y) < radius
        if near and not inside((bx + bw / 2, by + bh / 2), icon):
            candidates.append(box)
    if not candidates:
        return "", None
    icon_center = (x + w / 2, y + h / 2)
    box = min(candidates, key=lambda b: abs(b[1] + b[3] / 2 - icon_center[1]) + abs(b[0] - (x + w)) * 0.3)
    bx, by, bw, bh = box
    pad = 4
    crop = after[max(0, by - pad):by + bh + pad, max(0, bx - pad):bx + bw + pad]
    if crop.size == 0:
        return "", box
    words = ocr(crop)
    text = " ".join(w.text for w in sorted(words, key=lambda w: (w.box[1], w.box[0])))
    return clean_name(text), box


def clean_name(text: str, max_words: int = 4) -> str:
    """A tooltip reads as a label, not a sentence: keep the words, drop OCR debris."""
    import re

    words = []
    for raw in text.split():
        word = raw.strip("-–—·•:;,.()[]{}<>|/\\\"'`~*_=+")
        if not word or not re.search(r"[A-Za-z0-9]", word):
            continue
        if len(word) == 1 and not word.isalnum():
            continue
        words.append(word)
    while words and len(words[0]) == 1 and not words[0].isalpha():
        words.pop(0)
    return " ".join(words[:max_words])


def learn_from_tooltips(page, scene: PixelScene, ocr, memory: IconMemory, *, limit: int = 40, max_name_words: int = 4) -> TooltipLearning:
    """Hover every icon-only control in ``scene`` and remember what its tooltip says."""
    t0 = time.perf_counter()
    icons = [inf.control for inf in scene.controls if inf.rule.startswith("icon") and not inf.control.name]
    icons += [inf.control for inf in scene.controls if inf.rule.startswith("icon") and inf.control.name and inf.rule == "icon-unnamed"]
    learned: list[str] = []
    seen = 0
    image = _grab(page)
    for control in icons[:limit]:
        text, _ = read_tooltip(page, control.box, ocr)
        if not text:
            continue
        seen += 1
        name = " ".join(text.split()[:max_name_words])
        if len(name.split()) > max_name_words:
            continue
        if len(name) < 2:
            continue
        if memory.remember(image, control.box, name):
            learned.append(name)
    page.mouse.move(2, 2)  # park the pointer so it stops hovering
    return TooltipLearning(len(icons[:limit]), seen, len(learned), tuple(learned), round(time.perf_counter() - t0, 1))
