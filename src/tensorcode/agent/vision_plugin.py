"""A limited classifier adapter, not a scene-understanding model.

Structured interpretation retains every category alternative and its raw classifier
score. Each graph explicitly records that no scene structure was inferred.

The model is trained by ``eval/vision_hierarchy/train_concepts.py`` and loaded from
``$TENSORCODE_SCRATCH/vision/<name>.pickle``. Without it the plugin proposes nothing.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Iterable

from ..outcomes import Score
from ..records import Proposition, Ref
from .scene import SceneGraph, SceneProposal
from .plugin import Plugin


def model_path(name: str) -> Path:
    return Path(os.environ.get("TENSORCODE_SCRATCH", os.path.expanduser("~/.cache/tensorcode"))) / "vision" / f"{name}.pickle"


class VisionPlugin(Plugin):
    def __init__(self, name: str = "cifar10-concepts") -> None:
        super().__init__(name="vision")
        self.model_name = name
        self.model = None
        path = model_path(name)
        if path.exists():
            self.model = pickle.loads(path.read_bytes())
            self.kinds = {label: ("object",) for label in self.model["labels"]}

    def _distribution(self, image: Any):
        """Return validated label scores, without treating them as calibrated belief."""
        if self.model is None:
            return ()
        import numpy as np

        x = _as_array(image, self.model["size"])
        if x is None:
            return ()
        feats = self.model["hierarchy"].describe(x[None])
        raw = self.model["classifier"].predict_proba(self.model["scaler"].transform(feats))
        try:
            probs = np.asarray(raw, dtype=float)
        except (TypeError, ValueError):
            return ()
        labels = self.model["labels"]
        if (probs.ndim != 2 or probs.shape != (1, len(labels)) or not len(labels)
                or not all(isinstance(label, str) and label for label in labels)
                or len(set(labels)) != len(labels)
                or not np.isfinite(probs).all()
                or (probs < 0).any() or (probs > 1).any()
                or not np.isclose(probs[0].sum(), 1.0, rtol=1e-6, atol=1e-8)):
            return ()
        return tuple(zip(labels, (float(p) for p in probs[0])))

    def interpret_image(self, image: Any, ref: Ref) -> Iterable[SceneProposal]:
        """Propose the complete distribution of category alternatives.

        This whole-image model supplies no regions, spatial relationships, or holistic
        scene semantics. A shared entity identity lets alternatives disagree about its
        category without fabricating a different object for each label.
        """
        thing = Ref(f"{ref.id}/entity")
        for label, probability in self._distribution(image):
            yield SceneProposal(
                graph=SceneGraph(
                    image=ref,
                    nodes=(thing,),
                    propositions=(
                        Proposition("has_location", {"subject": thing, "object": ref}),
                        Proposition("is_a", {"subject": thing, "object": label}),
                    ),
                    limitations=("whole-image classification only; no scene structure inferred",),
                ),
                provenance=(f"vision:{self.model_name}", "classifier.predict_proba"),
                score=Score(probability, "uncalibrated", "classifier predict_proba; no calibration established"),
            )



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
