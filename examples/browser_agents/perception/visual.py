"""Visual segmentation provider: OCR + UI-element detector + classical CV rules.

Works on any screen image, with no DOM and no accessibility tree (see
``docs/revival/07-vision-perception.md`` for how it is built and what it scores). Every
element carries the rule that produced it and that rule's confidence, so fusion can prefer
a structural source and still take names from here.
"""

from __future__ import annotations

import time

from ..vision.perceive import PixelScene, perceive
from .protocol import Element, PerceivedScene, Region, Target, TextBlock, own


class VisionProvider:
    name = "vision"
    reliability = 0.6  # pixels are the fallback: right about what is on screen, weaker about structure

    def __init__(self, ocr=None, detector=None, icon_namer=None, text_candidates: bool = True) -> None:
        self._ocr, self._detector, self.icon_namer, self.text_candidates = ocr, detector, icon_namer, text_candidates

    def available(self) -> bool:
        try:
            import cv2  # noqa: F401
            import doctr  # noqa: F401
        except Exception:
            return False
        return True

    @property
    def ocr(self):
        if self._ocr is None:
            from ..vision.models import DoctrOCR

            self._ocr = DoctrOCR()
        return self._ocr

    @property
    def detector(self):
        if self._detector is None:
            from ..vision.models import IconDetector

            self._detector = IconDetector()
        return self._detector

    def perceive(self, target: Target) -> PerceivedScene:
        image = target.picture()
        if image is None:
            raise ValueError("the vision provider needs Target.image or Target.grab")
        t0 = time.perf_counter()
        scene = perceive(image, ocr=self.ocr, detector=self.detector, icon_namer=self.icon_namer, text_candidates=self.text_candidates)
        out = from_pixel_scene(scene)
        if target.image_origin != (0, 0):  # the image is a window crop: report in screen coordinates
            from .protocol import shift

            out = shift(out, *target.image_origin)
        out.timings_ms = {**scene.timings_ms, "total": (time.perf_counter() - t0) * 1e3}
        return out


def from_pixel_scene(scene: PixelScene) -> PerceivedScene:
    elements = []
    for inf in scene.controls:
        c = inf.control
        state = set()
        if c.checked:
            state.add("checked")
        if c.role == "textbox":
            state.add("editable")
        elements.append(Element(c.role, c.name, c.value, c.hint, c.section, c.box, c.point, frozenset(state),
                                round(inf.conf, 3), own("vision", "ocr+detector+cv", inf.rule, locator=f"box{c.box}")))
    texts = tuple(TextBlock(t.text, t.box, t.section, 0, round(conf, 3), own("vision", "ocr", "phrase")) for t, conf in scene.texts)
    regions = tuple(Region("window", w.title, w.box, round(w.conf, 3), own("vision", "frame-edges+title-strip", "complete" if w.complete else "partial")) for w in scene.windows)
    return PerceivedScene(tuple(elements), texts, regions, (), "pixels://screen", "", "", False, "vision")
