"""Names for icon-only controls, learned from examples (a visual vocabulary, not a model call).

Pixels alone cannot say that a terminal glyph in a dock means "Terminal". A person learns
it once (from a tooltip, a label, being told). ``IconMemory`` does the same: it stores a
small appearance vector per named example and answers with the nearest one above a
similarity threshold, or with nothing.

Examples can come from any labeled source: an accessibility tree on a machine where one
exists (how the evaluation here fits it: on the tuning split only), hover tooltips read
by OCR, or a person naming an icon. Appearance vectors are 24x24 RGB thumbnails, mean-
centered and L2-normalized; matching is cosine similarity. It recognizes the *same*
icon art again (other windows, other positions, other machines with that theme); it
does not generalize to a different icon set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

Box = tuple[int, int, int, int]


def appearance(rgb: np.ndarray, box: Box, size: int = 24) -> np.ndarray | None:
    import cv2

    x, y, w, h = box
    crop = rgb[max(0, y):y + h, max(0, x):x + w]
    if crop.shape[0] < 6 or crop.shape[1] < 6:
        return None
    crop = trim(crop)
    side = max(crop.shape[:2])  # pad to square so aspect ratio is kept
    pad = np.zeros((side, side, 3), np.uint8) + np.median(crop.reshape(-1, 3), axis=0).astype(np.uint8)
    oy, ox = (side - crop.shape[0]) // 2, (side - crop.shape[1]) // 2
    pad[oy:oy + crop.shape[0], ox:ox + crop.shape[1]] = crop
    v = cv2.resize(pad, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    v -= v.mean()
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-6 else None


def trim(crop: np.ndarray, tol: int = 30) -> np.ndarray:
    """Crop to the glyph: drop margins that match the crop's border color (boxes from different sources differ in padding)."""
    ring = np.concatenate([crop[0], crop[-1], crop[:, 0], crop[:, -1]]).astype(np.int16)
    bg = np.median(ring, axis=0)
    mask = np.abs(crop.astype(np.int16) - bg).sum(-1) > tol
    if mask.sum() < 12:
        return crop
    ys, xs = np.nonzero(mask)
    return crop[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


@dataclass
class IconMemory:
    threshold: float = 0.9
    vectors: list[np.ndarray] = field(default_factory=list)
    names: list[str] = field(default_factory=list)

    def remember(self, rgb: np.ndarray, box: Box, name: str) -> bool:
        v = appearance(rgb, box)
        if v is None or not name:
            return False
        if self.vectors:
            sims = np.stack(self.vectors) @ v
            i = int(sims.argmax())
            if sims[i] > 0.985 and self.names[i] == name:
                return False  # already known
        self.vectors.append(v)
        self.names.append(name)
        return True

    def __call__(self, rgb: np.ndarray, box: Box) -> tuple[str, float]:
        v = appearance(rgb, box)
        if v is None or not self.vectors:
            return "", 0.0
        sims = np.stack(self.vectors) @ v
        i = int(sims.argmax())
        return (self.names[i], float(sims[i])) if sims[i] >= self.threshold else ("", float(sims[i]))

    def save(self, path: Path) -> None:
        np.savez_compressed(path, vectors=np.stack(self.vectors) if self.vectors else np.zeros((0, 1728), np.float32), names=np.array(self.names), threshold=self.threshold)

    @classmethod
    def load(cls, path: Path) -> IconMemory:
        data = np.load(path)
        return cls(float(data["threshold"]), list(data["vectors"]), [str(n) for n in data["names"]])
