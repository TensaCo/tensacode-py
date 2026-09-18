"""Does a learned *hierarchy* of image features beat a flat one of the same size?

    python -m eval.vision_hierarchy.cifar

**Pre-registered, written before the first run (2026-09-18).**

Data: CIFAR-10 (Krizhevsky 2009; public), from ``~/.cache/tensorcode/datasets``.
Features are learned without labels on the 10,000 images of ``data_batch_1``; a linear
classifier (logistic regression, standardised features, C=1.0, fixed — nothing is tuned
on test) is trained on those same 10,000 images' labels and scored on the first 2,000
images of ``test_batch``, which no feature or classifier ever saw. Seed 0.

Arms, all describing an image with 1,024 numbers (the top map pooled into 2x2 regions):

==================  ========================================================================
flat                one layer: 6x6 patches, 256 k-means features
hier2               6x6 patches -> 64 features; pool 2; 3x3 of those -> 256 features
hier3               6x6 -> 64; pool 2; 3x3 -> 128; 3x3 -> 256
hier2_random_top    hier2 with its second layer's centroids left random (not learned)
==================  ========================================================================

Predictions:

1. every arm is well above chance (10%);
2. **hier2 beats flat by >= 2.0 points** (the claim that hierarchy helps at equal size);
3. **hier2 beats hier2_random_top by >= 2.0 points** (the claim that the upper layer's
   *learning* matters, not only its architecture);
4. hier3 is no worse than hier2 by more than 1.0 point (a third layer on 32x32 images
   may add little; this prediction is only that it does not break).

95% Wilson intervals on 2,000 test images are about +/-2 points, so a 2-point gap is
the smallest this test can resolve; smaller differences are reported, not claimed.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).parents[2] / "src")]

from tensorcode.vision.hierarchy import Hierarchy, Layer  # noqa: E402

DATA = Path.home() / ".cache" / "tensorcode" / "datasets" / "cifar-10-batches-py"
OUT = Path(__file__).parents[1] / "results" / "vision_hierarchy_cifar.json"


def load(name: str) -> tuple[np.ndarray, np.ndarray]:
    d = pickle.loads((DATA / name).read_bytes(), encoding="latin1")
    x = d["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return x, np.array(d["labels"])


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def arms() -> dict:
    return {
        "flat": (lambda: Hierarchy([Layer(6, 256)]), set()),
        "hier2": (lambda: Hierarchy([Layer(6, 64), Layer(3, 256, pool_before=2, eps=0.1)]), set()),
        "hier3": (lambda: Hierarchy([Layer(6, 64), Layer(3, 128, pool_before=2, eps=0.1), Layer(3, 256, eps=0.1)]), set()),
        "hier2_random_top": (lambda: Hierarchy([Layer(6, 64), Layer(3, 256, pool_before=2, eps=0.1)]), {1}),
    }


def main() -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    xtr, ytr = load("data_batch_1")
    xte, yte = load("test_batch")
    xte, yte = xte[:2000], yte[:2000]
    rows = {}
    for name, (make, random_layers) in arms().items():
        rng = np.random.default_rng(0)
        t0 = time.perf_counter()
        h = make()
        h.fit(xtr, rng, random_layers=random_layers)
        ftr, fte = h.describe(xtr), h.describe(xte)
        scaler = StandardScaler().fit(ftr)
        clf = LogisticRegression(C=1.0, max_iter=2000).fit(scaler.transform(ftr), ytr)
        correct = int((clf.predict(scaler.transform(fte)) == yte).sum())
        rows[name] = {"accuracy": round(correct / len(yte), 4), "ci95": wilson(correct, len(yte)), "dims": int(ftr.shape[1]),
                      "layers": [(l.size, l.k, l.pool_before) for l in h.layers], "seconds": round(time.perf_counter() - t0, 1)}
        print(name, rows[name], flush=True)
    a = {k: v["accuracy"] for k, v in rows.items()}
    verdict = {
        "above_chance": all(v > 0.15 for v in a.values()),
        "hier2_beats_flat_by_2pts": a["hier2"] - a["flat"] >= 0.02,
        "learning_top_beats_random_top_by_2pts": a["hier2"] - a["hier2_random_top"] >= 0.02,
        "hier3_not_worse_than_hier2_by_1pt": a["hier3"] >= a["hier2"] - 0.01,
    }
    print(verdict)
    args.out.write_text(json.dumps({"eval": "vision_hierarchy_cifar", "date": "2026-09-18", "seed": 0,
                                    "preregistered": "module docstring, committed before the first run",
                                    "data": "CIFAR-10 data_batch_1 (train, 10k) / test_batch[:2000]",
                                    "arms": rows, "verdict": verdict}, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
