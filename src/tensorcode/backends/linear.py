"""A small learned text classifier: TF-IDF features + multinomial logistic regression.

Requires the ``learned`` extra (scikit-learn, numpy). Abstention is governed by a
threshold chosen on held-out validation data to meet a target selective accuracy,
after temperature scaling on the same split. The fit report records what the
threshold is based on; nothing here is claimed beyond that split.
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion

from ..outcomes import Score, Unknown
from ..runtime import Output, Profile, Request, Traits


@dataclass(frozen=True)
class FitReport:
    train_size: int
    validation_size: int
    temperature: float
    threshold: float
    target_accuracy: float
    validation_coverage: float
    validation_selective_accuracy: float
    validation_accuracy: float
    fit_seconds: float
    basis: str


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


@dataclass
class LinearTextClassifier:
    labels: type[enum.Enum]
    features: FeatureUnion
    model: LogisticRegression
    temperature: float
    threshold: float
    basis: str
    name: str = "tfidf-logreg"
    version: str = "1"
    op: str = "classify"
    traits: Traits = Traits(locality="in_process", egress=False, deterministic=True, requires=frozenset({"sklearn"}))
    profile: Profile = field(default_factory=lambda: Profile(source="declared: in-process, no metered spend; latency unmeasured", usd_per_call=0.0))

    @classmethod
    def fit(
        cls,
        texts: Sequence[str],
        labels: Sequence[enum.Enum],
        *,
        label_type: type[enum.Enum],
        validation: tuple[Sequence[str], Sequence[enum.Enum]],
        target_accuracy: float,
        basis: str,
        seed: int = 0,
    ) -> tuple[LinearTextClassifier, FitReport]:
        t0 = time.perf_counter()
        features = FeatureUnion(
            [
                ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
                ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)),
            ]
        )
        x = features.fit_transform(texts)
        y = [label.value for label in labels]
        model = LogisticRegression(C=20.0, max_iter=2000, random_state=seed)
        model.fit(x, y)

        val_texts, val_labels = validation
        logits = model.decision_function(features.transform(val_texts))
        val_y = np.array([model.classes_.tolist().index(label.value) for label in val_labels])

        # temperature scaling: minimize validation NLL over a log grid
        grid = np.exp(np.linspace(np.log(0.05), np.log(5.0), 200))
        nll = [-np.log(_softmax(logits / t)[np.arange(len(val_y)), val_y] + 1e-12).mean() for t in grid]
        temperature = float(grid[int(np.argmin(nll))])
        probs = _softmax(logits / temperature)
        conf, pred = probs.max(axis=1), probs.argmax(axis=1)
        correct = pred == val_y

        # lowest threshold whose selective accuracy on validation meets the target
        order = np.argsort(-conf)
        cum_acc = np.cumsum(correct[order]) / np.arange(1, len(order) + 1)
        ok = np.nonzero(cum_acc >= target_accuracy)[0]
        k = int(ok.max()) + 1 if len(ok) else 0
        threshold = float(conf[order][k - 1]) if k else 1.0
        impl = cls(label_type, features, model, temperature, threshold, basis)
        report = FitReport(
            train_size=len(texts),
            validation_size=len(val_texts),
            temperature=temperature,
            threshold=threshold,
            target_accuracy=target_accuracy,
            validation_coverage=k / len(order),
            validation_selective_accuracy=float(cum_acc[k - 1]) if k else float("nan"),
            validation_accuracy=float(correct.mean()),
            fit_seconds=time.perf_counter() - t0,
            basis=basis,
        )
        return impl, report

    def accepts(self, request: Request) -> bool:
        return request.op == "classify" and request.target is self.labels and isinstance(request.subject, str)

    def probabilities(self, texts: Sequence[str]) -> np.ndarray:
        return _softmax(self.model.decision_function(self.features.transform(texts)) / self.temperature)

    def run(self, requests: Sequence[Request]) -> list[Output]:
        probs = self.probabilities([r.subject for r in requests])
        classes = self.model.classes_
        outs = []
        for row in probs:
            top = np.argsort(-row)[:3]
            cands = tuple((self.labels(classes[i]), Score(float(row[i]), "probability", self.basis)) for i in top)
            if row[top[0]] < self.threshold:
                outs.append(Output(Unknown("below_threshold", f"p={row[top[0]]:.3f} < {self.threshold:.3f}", cands)))
            else:
                outs.append(Output(cands[0][0], cands[0][1]))
        return outs
