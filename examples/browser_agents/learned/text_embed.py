"""A small learned text encoder for UI labels, and the ops that use it.

Labels are short, noisy and never seen exactly twice: ``Personnel no.`` in one render,
``Badge ID`` in the next, ``Personne1 no,`` when it comes back through OCR. Trigram
similarity handles the first two badly and the third by luck. This encoder maps a label
to a vector, trained so that phrasings of the same thing land together and OCR damage
does not move a label far.

Deliberately tiny: hashed character n-grams into an embedding bag, mean-pooled, one
hidden layer, L2-normalised output. It runs on CPU in microseconds, is deterministic for
a fixed seed and checkpoint, and needs no pretrained weights (none were cached, and
downloading a sentence encoder would have made the comparison against trigrams a
comparison of pretraining corpora instead of of the idea).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

_WORD = re.compile(r"[a-z0-9]+")


def normalize(text: str) -> str:
    return " ".join(_WORD.findall(text.lower()))


def features(text: str, *, buckets: int, ngrams: tuple[int, ...] = (3, 4, 5)) -> list[int]:
    """Hashed character n-grams of the padded, normalised label, plus whole words."""
    s = f" {normalize(text)} "
    out: list[int] = []
    for n in ngrams:
        for i in range(max(0, len(s) - n + 1)):
            out.append(int(hashlib.blake2b(s[i : i + n].encode(), digest_size=8).hexdigest(), 16) % buckets)
    for word in s.split():
        out.append(int(hashlib.blake2b(f"w:{word}".encode(), digest_size=8).hexdigest(), 16) % buckets)
    return out or [0]


@dataclass
class LabelEncoder:
    """Hashed n-gram embedding bag -> mean pool -> tanh layer -> unit vector."""

    buckets: int
    dim: int
    emb: np.ndarray  # (buckets, dim)
    w: np.ndarray  # (dim, dim)
    b: np.ndarray  # (dim,)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        rows = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            idx = features(t, buckets=self.buckets)
            rows[i] = self.emb[idx].mean(axis=0)
        h = np.tanh(rows @ self.w + self.b)
        norms = np.linalg.norm(h, axis=1, keepdims=True)
        return h / np.maximum(norms, 1e-8)

    def similarity(self, query: str, name: str) -> float:
        a, b = self.encode([query, name])
        return float(np.dot(a, b))

    # -- persistence

    def save(self, path: Path) -> None:
        np.savez(path, emb=self.emb, w=self.w, b=self.b, meta=np.array([self.buckets, self.dim]))

    @classmethod
    def load(cls, path: Path) -> LabelEncoder:
        z = np.load(path)
        buckets, dim = (int(v) for v in z["meta"])
        return cls(buckets, dim, z["emb"], z["w"], z["b"])


# ------------------------------------------------------------------ the ops


@dataclass
class LearnedLabelMatcher:
    """``rank`` over controls by learned label similarity. Same contract as LabelMatcher."""

    encoder: LabelEncoder
    source: str
    name: str = "learned-label-matcher"
    version: str = "1"
    op: str = "rank"
    latency_ms_p50: float | None = None
    latency_ms_p95: float | None = None

    @property
    def traits(self):  # imported lazily so this module is importable without the browser stack
        import tensorcode as tc

        return tc.Traits(locality="in_process", egress=False, deterministic=True, requires=frozenset({"numpy"}))

    @property
    def profile(self):
        import tensorcode as tc

        return tc.Profile(source=self.source, latency_ms_p50=self.latency_ms_p50, latency_ms_p95=self.latency_ms_p95, usd_per_call=0.0)

    def accepts(self, request) -> bool:
        from ..browser import Control

        return request.op == "rank" and isinstance(request.subject, tuple) and all(isinstance(c, Control) for c in request.params.get("candidates", ()))

    def run(self, requests):
        import tensorcode as tc

        outs = []
        for r in requests:
            candidates = r.params["candidates"]
            names = [f"{c.name} {c.group}".strip() for c in candidates]
            if not names:
                outs.append(tc.Output(()))
                continue
            qv = self.encoder.encode(list(r.subject))
            cv = self.encoder.encode(names)
            best = (cv @ qv.T).max(axis=1)  # best over the query's synonyms
            scored = [(c, tc.Score(float(s), "learned-similarity", self.source)) for c, s in zip(candidates, best)]
            outs.append(tc.Output(sorted(scored, key=lambda p: p[1].value, reverse=True)))
        return outs


@dataclass
class ConceptClassifier:
    """``classify`` a UI label as one of the form concepts, with a calibrated abstention."""

    encoder: LabelEncoder
    prototypes: dict[str, np.ndarray]  # concept -> unit vector (mean of its training phrasings)
    threshold: float
    temperature: float
    source: str
    name: str = "learned-concept-classifier"
    version: str = "1"
    op: str = "classify"
    report: dict = field(default_factory=dict)

    def scores(self, label: str) -> dict[str, float]:
        v = self.encoder.encode([label])[0]
        return {c: float(np.dot(v, p)) for c, p in self.prototypes.items()}

    def probabilities(self, label: str) -> dict[str, float]:
        z = {c: s / self.temperature for c, s in self.scores(label).items()}
        m = max(z.values())
        e = {c: math.exp(v - m) for c, v in z.items()}
        total = sum(e.values())
        return {c: v / total for c, v in e.items()}

    def save(self, path: Path) -> None:
        self.encoder.save(path.with_suffix(".encoder.npz"))
        path.write_text(json.dumps({
            "prototypes": {c: v.tolist() for c, v in self.prototypes.items()},
            "threshold": self.threshold, "temperature": self.temperature, "source": self.source, "report": self.report,
        }, indent=1))

    @classmethod
    def load(cls, path: Path) -> ConceptClassifier:
        meta = json.loads(path.read_text())
        return cls(
            LabelEncoder.load(path.with_suffix(".encoder.npz")),
            {c: np.array(v, dtype=np.float32) for c, v in meta["prototypes"].items()},
            meta["threshold"], meta["temperature"], meta["source"], report=meta.get("report", {}),
        )
