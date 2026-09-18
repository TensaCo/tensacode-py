"""Learn a control's role from pixels, with the DOM as a label nobody typed.

    PYTHONPATH=src:. python eval/training/train_role_model.py --data DIR --cache DIR --out DIR

A pixel perceiver has to say what a rectangle *is* — a button, a text field, a checkbox,
a tab, a table header. The hand-written rules in the vision path get the role right for
0.15-0.17 of matched textboxes on held-out frames, which is the weakest number in that
report. The label is free: the DOM says the role at the instant the screenshot was taken.

Features are what a pixel provider actually has: the box's geometry and position, the
recognizer's words inside it (how many, how confident, digits, length), the detector's
opinion, and the local density of other elements. No DOM feature is used as an input —
only as the label.

Reported per class, because the classes are wildly unbalanced (10,306 buttons against
430 textboxes in the captured frames), so overall accuracy is a majority-class number
and says nothing about the roles that matter.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).parents[2] / "src"), str(Path(__file__).parents[2])]

ROLES = ("button", "textbox", "checkbox", "tab", "combobox", "columnheader")
FEATURES = (
    "width", "height", "aspect", "area", "x_frac", "y_frac", "words", "chars", "mean_conf",
    "digit_frac", "upper_frac", "has_colon", "detector_iou", "neighbours_row", "neighbours_col", "words_per_area",
)


def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    return inter / (aw * ah + bw * bh - inter)


def frame_rows(frame: dict, cache_blob: dict, size: tuple[int, int] = (1280, 800)) -> list[tuple[list[float], str]]:
    words = [w for w in (cache_blob.get("words") or []) if getattr(w, "text", "").strip()]
    icons = cache_blob.get("icons") or []
    controls = [c for c in frame["screen"]["controls"] if c.get("role") in ROLES and not c.get("disabled")]
    boxes = [tuple(c["box"]) for c in controls]
    rows = []
    for control, box in zip(controls, boxes):
        x, y, w, h = box
        if w < 4 or h < 4:
            continue
        inside = [word for word in words if x <= word.box[0] + word.box[2] / 2 <= x + w and y <= word.box[1] + word.box[3] / 2 <= y + h]
        text = " ".join(word.text for word in inside)
        chars = len(text)
        digits = sum(ch.isdigit() for ch in text)
        uppers = sum(ch.isupper() for ch in text)
        det = max((_iou(box, tuple(b)) for b, _ in icons), default=0.0)
        same_row = sum(1 for bx, by, bw, bh in boxes if abs(by - y) < h / 2 and bx != x)
        same_col = sum(1 for bx, by, bw, bh in boxes if abs(bx - x) < w / 2 and by != y)
        rows.append(([
            w, h, w / max(h, 1), w * h / 1e4, x / size[0], y / size[1],
            len(inside), chars, float(np.mean([word.conf for word in inside])) if inside else 0.0,
            digits / max(chars, 1), uppers / max(chars, 1), float(":" in text),
            det, same_row, same_col, len(inside) / max(w * h / 1e4, 1e-3),
        ], control["role"]))
    return rows


def load_split(data: Path, cache: Path, splits: tuple[str, ...]) -> tuple[np.ndarray, list[str]]:
    x, y = [], []
    for meta_path in sorted(data.glob("*.json")):
        frame = json.loads(meta_path.read_text())
        if frame["meta"].get("split") not in splits:
            continue
        blob = cache / f"{meta_path.stem}.pkl"
        if not blob.exists():
            continue
        for features, role in frame_rows(frame, pickle.loads(blob.read_bytes())):
            x.append(features)
            y.append(role)
    return np.array(x, dtype=np.float32), y


def report(name: str, y_true: list[str], y_pred: list[str]) -> dict:
    classes = sorted(set(y_true) | set(y_pred))
    per_class = {}
    for c in classes:
        tp = sum(t == c and p == c for t, p in zip(y_true, y_pred))
        fp = sum(t != c and p == c for t, p in zip(y_true, y_pred))
        fn = sum(t == c and p != c for t, p in zip(y_true, y_pred))
        per_class[c] = {
            "support": sum(t == c for t in y_true),
            "recall": round(tp / (tp + fn), 4) if tp + fn else None,
            "precision": round(tp / (tp + fp), 4) if tp + fp else None,
        }
    return {
        "split": name, "n": len(y_true),
        "accuracy": round(sum(t == p for t, p in zip(y_true, y_pred)) / max(len(y_true), 1), 4),
        "majority_baseline": round(Counter(y_true).most_common(1)[0][1] / max(len(y_true), 1), 4),
        "macro_recall": round(float(np.mean([v["recall"] for v in per_class.values() if v["recall"] is not None])), 4),
        "per_class": per_class,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from sklearn.ensemble import HistGradientBoostingClassifier

    xtr, ytr = load_split(args.data, args.cache, ("tune",))
    if not len(xtr):
        raise SystemExit("no training rows: check --data/--cache")
    t0 = time.perf_counter()
    model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=args.seed, class_weight="balanced")
    model.fit(xtr, ytr)
    train_seconds = time.perf_counter() - t0

    out = {"features": FEATURES, "train_rows": len(ytr), "train_seconds": round(train_seconds, 1),
           "train_class_counts": dict(Counter(ytr)), "seed": args.seed, "splits": {}}
    for split in ("tune", "test_app", "test_os"):
        x, y = load_split(args.data, args.cache, (split,))
        if not len(x):
            continue
        t0 = time.perf_counter()
        pred = list(model.predict(x))
        per_item_ms = (time.perf_counter() - t0) * 1000 / len(x)
        out["splits"][split] = {**report(split, y, pred), "predict_ms_per_element": round(per_item_ms, 4)}

    with open(args.out / "role_model.pkl", "wb") as fh:
        pickle.dump(model, fh)
    (args.out / "role_model.report.json").write_text(json.dumps(out, indent=1))
    for split, s in out["splits"].items():
        print(f"{split:<9} n={s['n']:<6} acc={s['accuracy']:.3f} (majority {s['majority_baseline']:.3f})  macro-recall={s['macro_recall']:.3f}")
        for c, v in sorted(s["per_class"].items()):
            if v["support"]:
                print(f"    {c:<13} support={v['support']:<5} recall={v['recall']}  precision={v['precision']}")


if __name__ == "__main__":
    main()
