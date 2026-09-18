"""A plugin that sees: learned hierarchical features plus a concept classifier over them.

Given an image, it reports what it recognises as claims — ``thing has_location image``
and ``thing is_a <concept>`` — with the classifier's probability as evidence. Below the
abstention threshold (chosen on held-out data, recorded with the model) it reports
nothing rather than a guess. Its vocabulary is its concept labels: the words it was
taught with, not words it was handed for a demo.

The model is trained by ``eval/vision_hierarchy/train_concepts.py`` and loaded from
``$TENSORCODE_SCRATCH/vision/<name>.pickle``. Without it the plugin sees nothing.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Iterable

from ..records import Claim, Ref
from .plugin import Plugin


def model_path(name: str) -> Path:
    return Path(os.environ.get("TENSORCODE_SCRATCH", os.path.expanduser("~/.cache/tensorcode"))) / "vision" / f"{name}.pickle"


class VisionPlugin(Plugin):
    def __init__(self, name: str = "cifar10-concepts") -> None:
        super().__init__(name="vision")
        self.model = None
        path = model_path(name)
        if path.exists():
            self.model = pickle.loads(path.read_bytes())
            self.kinds = {label: ("object",) for label in self.model["labels"]}

    def see(self, image: Any, ref: Any) -> Iterable[Claim]:
        if self.model is None:
            return
        import numpy as np

        x = _as_array(image, self.model["size"])
        if x is None:
            return
        feats = self.model["hierarchy"].describe(x[None])
        probs = self.model["classifier"].predict_proba(self.model["scaler"].transform(feats))[0]
        best = int(np.argmax(probs))
        if probs[best] < self.model["threshold"]:
            return  # not sure enough to say; the store stays silent rather than wrong
        label = self.model["labels"][best]
        thing = Ref(f"{ref.id}/{label}")
        yield Claim(thing, "has_location", ref)
        yield Claim(thing, "is_a", label)

    def display(self, ref: Any) -> str:
        rid = getattr(ref, "id", str(ref))
        if rid.startswith("image:") and "/" in rid:
            label = rid.rsplit("/", 1)[-1]
            return ("an " if label[:1] in "aeiou" else "a ") + label
        return super().display(ref)


def _as_array(image: Any, size: int):
    """An image as (size, size, 3) uint8: from an array, a path, or bytes."""
    import numpy as np

    if isinstance(image, np.ndarray):
        arr = image
    else:
        try:
            from PIL import Image
        except ImportError:
            return None
        import io

        src = Image.open(io.BytesIO(image)) if isinstance(image, (bytes, bytearray)) else Image.open(image)
        arr = np.asarray(src.convert("RGB").resize((size, size)))
    if arr.shape[:2] != (size, size):
        try:
            from PIL import Image

            arr = np.asarray(Image.fromarray(arr.astype("uint8")).convert("RGB").resize((size, size)))
        except ImportError:
            return None
    return arr.astype("uint8")
