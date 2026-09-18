"""Train the vision plugin's concept model: hierarchical features + a linear concept classifier.

    python -m eval.vision_hierarchy.train_concepts

Splits (CIFAR-10), none reused across roles:

* features learned (unsupervised) on 10,000 images of ``data_batch_1``;
* classifier trained on ``data_batch_1..4`` (40,000 labelled images);
* abstention threshold chosen on ``data_batch_5`` (10,000): the lowest probability at
  which answered items reach **0.90 precision** — the plugin says nothing below it;
* reported on ``test_batch[2000:4000]``, which the pre-registered hierarchy test
  (``cifar.py``, ``test_batch[:2000]``) did not use.

The architecture is chosen on the **validation** batch (``data_batch_5``) between
``flat`` and ``hier2``: the pre-registered test (``cifar.py``) found the flat layer
better (62.7% vs 57.1% on ``test_batch[:2000]``), but that is a test result, so it is
not used to choose; the choice is re-made on validation data here. The model is written to ``$TENSORCODE_SCRATCH/vision/cifar10-concepts.pickle``
and a summary is appended to ``eval/results/vision_concepts.jsonl``.
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).parents[2] / "src"), str(Path(__file__).parents[2])]

from eval.vision_hierarchy.cifar import load  # noqa: E402
from tensorcode.agent.vision_plugin import model_path  # noqa: E402
from tensorcode.vision.hierarchy import Hierarchy, Layer  # noqa: E402

LABELS = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
OUT = Path(__file__).parents[1] / "results" / "vision_concepts.jsonl"


def threshold_for(probs: np.ndarray, y: np.ndarray, target: float = 0.90) -> float:
    conf = probs.max(1)
    right = probs.argmax(1) == y
    for t in np.unique(np.round(conf, 3)):
        answered = conf >= t
        if answered.sum() >= 50 and right[answered].mean() >= target:
            return float(t)
    return 1.01


def main() -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(0)
    parts = [load(f"data_batch_{i}") for i in range(1, 5)]
    xtr, ytr = np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])
    xcal, ycal = load("data_batch_5")
    xte, yte = load("test_batch")
    xte, yte = xte[2000:4000], yte[2000:4000]
    t0 = time.perf_counter()
    candidates = {"flat": [Layer(6, 256)], "hier2": [Layer(6, 64), Layer(3, 256, pool_before=2, eps=0.1)]}
    chosen = None
    for name, layers in candidates.items():
        h_ = Hierarchy(layers)
        h_.fit(xtr[:10000], np.random.default_rng(0))
        f_ = h_.describe(xtr)
        sc_ = StandardScaler().fit(f_)
        clf_ = LogisticRegression(C=1.0, max_iter=3000).fit(sc_.transform(f_), ytr)
        pcal_ = clf_.predict_proba(sc_.transform(h_.describe(xcal)))
        acc = float((pcal_.argmax(1) == ycal).mean())
        print(f"validation {name}: {acc:.4f}", flush=True)
        if chosen is None or acc > chosen[0]:
            chosen = (acc, name, h_, sc_, clf_, pcal_)
    val_acc, arch, h, scaler, clf, pcal = chosen
    del rng
    t = threshold_for(pcal, ycal)
    pte = clf.predict_proba(scaler.transform(h.describe(xte)))
    conf, pred = pte.max(1), pte.argmax(1)
    answered = conf >= t
    summary = {"eval": "vision_concepts", "at": time.strftime("%Y-%m-%dT%H:%M:%S"), "model": "cifar10-concepts",
               "architecture": arch, "validation_accuracy": round(val_acc, 4),
               "threshold": round(t, 3), "test_accuracy_all": round(float((pred == yte).mean()), 4),
               "test_coverage": round(float(answered.mean()), 4),
               "test_precision_when_answering": round(float((pred[answered] == yte[answered]).mean()), 4) if answered.any() else None,
               "splits": "features: batch1[:10k]; classifier: batch1-4; threshold: batch5; test: test_batch[2000:4000]",
               "seconds": round(time.perf_counter() - t0, 1)}
    path = model_path("cifar10-concepts")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps({"hierarchy": h, "scaler": scaler, "classifier": clf, "labels": LABELS,
                                   "threshold": t, "size": 32, "summary": summary}))
    with OUT.open("a") as f:
        f.write(json.dumps(summary) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
