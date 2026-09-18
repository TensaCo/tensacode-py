"""A learned ``choose`` implementation: rank intentions with weights fitted to our own runs.

Registered like any other implementation, so the runtime routes to it and the trace counts
it. It abstains when the top two options are within a margin, which is what the
hand-written chooser does, so a tie is still a tie rather than a coin flip.

The weights come from ``eval/training/train_intention_ranker.py`` and are fitted WITHOUT
the intention's ``priority`` field, since that field is the hand-written objective itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass
class LearnedChooser:
    w: np.ndarray
    encoder: object | None  # LabelEncoder, or None to run without text features
    source: str
    margin: float = 0.0
    name: str = "learned-intention-ranker"
    version: str = "1"
    op: str = "choose"
    latency_ms_p50: float | None = None
    latency_ms_p95: float | None = None
    calls: int = field(default=0, compare=False)

    @property
    def traits(self):
        import tensacode as tc

        return tc.Traits(locality="in_process", egress=False, deterministic=True, requires=frozenset({"numpy"}))

    @property
    def profile(self):
        import tensacode as tc

        return tc.Profile(source=self.source, latency_ms_p50=self.latency_ms_p50, latency_ms_p95=self.latency_ms_p95, usd_per_call=0.0)

    def accepts(self, request) -> bool:
        return request.op == "choose"

    def scores(self, options: Sequence[object]) -> np.ndarray:
        from eval.training.train_intention_ranker import featurize
        from eval.training.rollout_decisions import describe_intention

        n = len(options)
        x = np.stack([featurize(describe_intention(o), index=i, n=n, encoder=self.encoder) for i, o in enumerate(options)])
        return x @ self.w

    def run(self, requests):
        import tensacode as tc

        outs = []
        for r in requests:
            options = list(r.subject)
            self.calls += 1
            scores = self.scores(options)
            order = np.argsort(-scores)
            cands = tuple((options[i], tc.Score(float(scores[i]), "learned-utility", self.source)) for i in order)
            if len(order) > 1 and float(scores[order[0]] - scores[order[1]]) <= self.margin:
                outs.append(tc.Output(tc.Unknown("tie_within_margin", f"top scores {scores[order[0]]:.3g} vs {scores[order[1]]:.3g}", cands)))
            else:
                outs.append(tc.Output(options[order[0]], cands[0][1]))
        return outs

    @classmethod
    def load(cls, path: Path, *, encoder=None, source: str = "", margin: float = 0.0) -> LearnedChooser:
        z = np.load(path)
        return cls(z["w"], encoder, source or str(path), margin)
