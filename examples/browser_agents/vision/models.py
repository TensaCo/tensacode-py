"""Small local vision models behind two narrow interfaces (no LLM/VLM anywhere).

* ``DoctrOCR``: words with boxes and confidences (docTR: FAST-base text detector + CRNN
  MobileNetV3-large recognizer, ~60 MB of weights, runs on CUDA when available). Chosen
  over crnn_vgg16_bn / parseq / vitstr_small on the tuning frames: within 1 point of
  word recall at 35-60% of their latency (see docs/revival/07-vision-perception.md).
* ``IconDetector``: interactable-element boxes (OmniParser v2 ``icon_detect``, a YOLOv8
  model fine-tuned on UI screenshots, ~41 MB; note its AGPL license).

Both load lazily, report load time and memory, and can be freed. Anything with the same
call signature can stand in (tests use fakes; see ``perceive.py``).
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Word:
    text: str
    box: tuple[int, int, int, int]  # x, y, w, h in screenshot pixels
    conf: float
    line: int = -1  # the recognizer's line grouping, when it has one


@dataclass
class LoadStats:
    name: str
    seconds: float = 0.0
    cuda_mb: float = 0.0
    weights_mb: float = 0.0


def _cuda_mb() -> float:
    try:
        import torch

        return torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0
    except ImportError:  # pragma: no cover - torch is optional
        return 0.0


class DoctrOCR:
    def __init__(self, det_arch: str = "fast_base", reco_arch: str = "crnn_mobilenet_v3_large", device: str | None = None, reco_weights: str | None = None) -> None:
        self.det_arch, self.reco_arch, self.device, self.reco_weights = det_arch, reco_arch, device, reco_weights
        self.model = None
        self.stats = LoadStats(f"doctr:{det_arch}+{reco_arch}")

    def load(self) -> None:
        if self.model is not None:
            return
        import torch
        from doctr.models import ocr_predictor

        before, t0 = _cuda_mb(), time.perf_counter()
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ocr_predictor(det_arch=self.det_arch, reco_arch=self.reco_arch, pretrained=True, assume_straight_pages=True, preserve_aspect_ratio=True, symmetric_pad=True).to(device).eval()
        if self.reco_weights:  # e.g. fine-tuned on the fonts this agent actually reads (free labels: it typed the text)
            state = torch.load(self.reco_weights, map_location=device)
            self.model.reco_predictor.model.load_state_dict(state)
            self.model.reco_predictor.model.to(device).eval()
            self.stats.name += f"+finetune:{self.reco_weights.rsplit('/', 1)[-1]}"
        self.stats.seconds, self.stats.cuda_mb = time.perf_counter() - t0, _cuda_mb() - before
        self.stats.weights_mb = sum(p.numel() * p.element_size() for p in self.model.parameters()) / 1e6

    def __call__(self, rgb: np.ndarray) -> list[Word]:
        import torch

        self.load()
        h, w = rgb.shape[:2]
        with torch.inference_mode():
            doc = self.model([rgb])
        words: list[Word] = []
        n = 0
        for block in doc.pages[0].blocks:
            for line in block.lines:
                for word in line.words:
                    (x0, y0), (x1, y1) = word.geometry
                    box = (round(x0 * w), round(y0 * h), max(1, round((x1 - x0) * w)), max(1, round((y1 - y0) * h)))
                    words.append(Word(word.value, box, float(word.confidence), n))
                n += 1
        return words

    def free(self) -> None:
        self.model = None
        _collect()


class IconDetector:
    REPO, FILE = "microsoft/OmniParser-v2.0", "icon_detect/model.pt"

    def __init__(self, conf: float = 0.05, iou: float = 0.3, device: str | None = None) -> None:
        self.conf, self.iou, self.device = conf, iou, device
        self.model = None
        self.stats = LoadStats("omniparser-v2:icon_detect")

    def load(self) -> None:
        if self.model is not None:
            return
        import os

        import torch
        from huggingface_hub import hf_hub_download
        from ultralytics import YOLO

        before, t0 = _cuda_mb(), time.perf_counter()
        path = hf_hub_download(self.REPO, self.FILE)
        self.model = YOLO(path)
        self.model.to(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.stats.seconds, self.stats.cuda_mb, self.stats.weights_mb = time.perf_counter() - t0, _cuda_mb() - before, os.path.getsize(path) / 1e6

    def __call__(self, rgb: np.ndarray) -> list[tuple[tuple[int, int, int, int], float]]:
        self.load()
        result = self.model.predict(rgb[:, :, ::-1], imgsz=max(rgb.shape[:2]) // 32 * 32, conf=self.conf, iou=self.iou, verbose=False)[0]
        out = []
        for (x0, y0, x1, y1), c in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist()):
            out.append(((round(x0), round(y0), max(1, round(x1 - x0)), max(1, round(y1 - y0))), float(c)))
        return out

    def free(self) -> None:
        self.model = None
        _collect()


def _collect() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:  # pragma: no cover
        pass
