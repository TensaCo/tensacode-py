"""Train the UI-label encoder, and hold out phrasings nobody wrote for the purpose.

    PYTHONPATH=src:. python eval/training/train_label_encoder.py --out DIR [--epochs 40]

Supervision is free and mechanically true:

* the form concepts the access agent already knows, each with the phrasings its
  hand-written synonym list contains (``TEXT_CONCEPTS``);
* OCR damage: labels read back from pixels in the captured frames, paired with the DOM
  label at the same instant (``$SP/vision_data`` + the cached recognizer output), so a
  clean label and its damaged reading are a positive pair nobody annotated;
* control names from those frames as negatives, which is what a matcher must not
  confuse a field label with.

Held out and never trained on: the fourth label variant of each field in
``web/access.html`` (``Legal name``, ``Badge ID``, ``Effective from``,
``Supervisor e-mail``). The hand-written synonym lists contain three of the app's four
variants per field, so the fourth is a natural generalisation test: it exists because
the app's author wrote it, not because an evaluator chose it.

Objective: InfoNCE over in-batch negatives on those positive pairs, plus a margin term
pushing a label away from the other concepts' phrasings. Trained with plain numpy and a
fixed seed, so a rerun reproduces the checkpoint bit for bit.
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).parents[2] / "src"), str(Path(__file__).parents[2])]

from examples.browser_agents.learned.text_embed import LabelEncoder, features, normalize  # noqa: E402
from examples.browser_agents.tasks.access_script import TEXT_CONCEPTS  # noqa: E402

#: the app's own label table (web/access.html). The last of each list is held out.
APP_LABELS = {
    "name": ["Full name", "Employee name", "Name of new hire", "Legal name"],
    "employee_id": ["Employee ID", "Staff number", "Personnel no.", "Badge ID"],
    "start_date": ["Start date", "First day", "Access begins", "Effective from"],
    "manager": ["Manager email", "Approver email", "Reports to (email)", "Supervisor e-mail"],
}
HELD_OUT = {c: labels[-1] for c, labels in APP_LABELS.items()}

#: surface noise a label picks up between renders, and on its way through a recognizer
TYPO_MAP = str.maketrans({"o": "0", "l": "1", "i": "1", "e": "c", "s": "5", "a": "@"})


def augment(text: str, rng: random.Random) -> str:
    """One surface variant: case, punctuation, decoration, or a single character slip."""
    choice = rng.randrange(6)
    if choice == 0:
        return text.upper() if rng.random() < 0.5 else text.lower()
    if choice == 1:
        return text.replace(" ", rng.choice(["  ", " - ", "_"]))
    if choice == 2:
        return rng.choice(["{} *", "{} (required)", "Enter {}", "{}:", "* {}"]).format(text)
    if choice == 3:  # a single character slips, the way OCR slips
        i = rng.randrange(len(text))
        return text[:i] + text[i].translate(TYPO_MAP) + text[i + 1 :]
    if choice == 4:  # a character is dropped
        i = rng.randrange(len(text))
        return text[:i] + text[i + 1 :]
    words = text.split()
    return " ".join(words[:-1]) if len(words) > 2 else text  # truncated label


def ocr_pairs(data: Path, cache: Path, splits: tuple[str, ...]) -> list[tuple[str, str]]:
    """(DOM label, recognizer reading at the same place) from the captured frames."""
    pairs: list[tuple[str, str]] = []
    for meta_path in sorted(data.glob("*.json")):
        frame = json.loads(meta_path.read_text())
        if frame["meta"].get("split") not in splits:
            continue
        blob = cache / f"{meta_path.stem}.pkl"
        if not blob.exists():
            continue
        out = pickle.loads(blob.read_bytes())
        words = out.get("words") if isinstance(out, dict) else None
        if not words:
            continue
        read = [(w.text, tuple(w.box)) for w in words if getattr(w, "text", "").strip() and getattr(w, "box", None)]
        for control in frame["screen"]["controls"]:
            name = (control.get("name") or "").strip()
            if not name or len(name) > 40:
                continue
            x, y, w, h = control["box"]
            inside = [t for t, (bx, by, bw, bh) in read if bx + bw / 2 >= x and bx + bw / 2 <= x + w and by + bh / 2 >= y and by + bh / 2 <= y + h]
            if not inside:
                continue
            seen = " ".join(inside)
            if normalize(seen) and normalize(seen) != normalize(name):
                pairs.append((name, seen))
    return pairs


def negatives_from_frames(data: Path, splits: tuple[str, ...], limit: int = 4000) -> list[str]:
    names: list[str] = []
    for meta_path in sorted(data.glob("*.json")):
        frame = json.loads(meta_path.read_text())
        if frame["meta"].get("split") not in splits:
            continue
        for control in frame["screen"]["controls"]:
            name = (control.get("name") or "").strip()
            if name and len(name) <= 40:
                names.append(name)
    rng = random.Random(0)
    rng.shuffle(names)
    return names[:limit]


def build_pairs(data: Path | None, cache: Path | None, seed: int) -> tuple[list[tuple[str, str]], dict]:
    """Positive pairs (two surfaces of the same thing) and a manifest of where they came from."""
    rng = random.Random(seed)
    pairs: list[tuple[str, str]] = []
    counts = {"concept_phrasings": 0, "augmented": 0, "ocr": 0}

    # phrasings of one concept are surfaces of the same thing
    for concept, synonyms in TEXT_CONCEPTS.items():
        known = [s for s in synonyms] + [lab for lab in APP_LABELS[concept][:-1]]
        known = sorted({normalize(k): k for k in known}.values())
        for i, a in enumerate(known):
            for b in known[i + 1 :]:
                pairs.append((a, b))
                counts["concept_phrasings"] += 1
            for _ in range(12):
                pairs.append((a, augment(a, rng)))
                counts["augmented"] += 1

    if data is not None and cache is not None and data.exists():
        # the captured frames include access pages, so a held-out variant can appear there
        held = {normalize(v) for v in HELD_OUT.values()}
        clean = lambda text: normalize(text) not in held  # noqa: E731
        found = [(a, b) for a, b in ocr_pairs(data, cache, ("tune",)) if clean(a) and clean(b)]
        pairs += found
        counts["ocr"] = len(found)
        for name in negatives_from_frames(data, ("tune",), limit=1500):
            if not clean(name):
                continue
            for _ in range(2):
                extra = augment(name, rng)
                if clean(extra):
                    pairs.append((name, extra))
                    counts["augmented"] += 1
    return pairs, counts


def train(pairs: list[tuple[str, str]], *, buckets: int, dim: int, epochs: int, batch: int, lr: float, seed: int) -> tuple[LabelEncoder, list[float]]:
    rng = np.random.default_rng(seed)
    emb = (rng.standard_normal((buckets, dim)) * 0.05).astype(np.float32)
    w = (rng.standard_normal((dim, dim)) / np.sqrt(dim)).astype(np.float32)
    b = np.zeros(dim, dtype=np.float32)
    feats = [(features(a, buckets=buckets), features(c, buckets=buckets)) for a, c in pairs]
    losses = []

    def forward(idx_lists: list[list[int]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        bag = np.zeros((len(idx_lists), dim), dtype=np.float32)
        for i, idx in enumerate(idx_lists):
            bag[i] = emb[idx].mean(axis=0)
        pre = bag @ w + b
        h = np.tanh(pre)
        norm = np.maximum(np.linalg.norm(h, axis=1, keepdims=True), 1e-8)
        return bag, h / norm, (pre, h, norm)

    order = np.arange(len(feats))
    for epoch in range(epochs):
        rng.shuffle(order)
        total, n = 0.0, 0
        for start in range(0, len(order) - batch + 1, batch):
            rows = order[start : start + batch]
            left = [feats[i][0] for i in rows]
            right = [feats[i][1] for i in rows]
            bag_l, zl, (pre_l, h_l, norm_l) = forward(left)
            bag_r, zr, (pre_r, h_r, norm_r) = forward(right)
            logits = (zl @ zr.T) / 0.07
            m = logits.max(axis=1, keepdims=True)
            p = np.exp(logits - m)
            p /= p.sum(axis=1, keepdims=True)
            target = np.eye(len(rows), dtype=np.float32)
            loss = float(-np.log(np.maximum(np.diagonal(p), 1e-12)).mean())
            total += loss
            n += 1

            # gradients of InfoNCE (symmetric in the two views)
            g_logits = (p - target) / len(rows)
            g_zl = (g_logits @ zr) / 0.07
            g_zr = (g_logits.T @ zl) / 0.07

            def backward(g_z, z, pre, h, norm, bag, idx_lists):
                g_h = (g_z - z * (g_z * z).sum(axis=1, keepdims=True)) / norm
                g_pre = g_h * (1 - np.tanh(pre) ** 2)
                g_w = bag.T @ g_pre
                g_b = g_pre.sum(axis=0)
                g_bag = g_pre @ w.T
                return g_w, g_b, g_bag

            gw1, gb1, gbag1 = backward(g_zl, zl, pre_l, h_l, norm_l, bag_l, left)
            gw2, gb2, gbag2 = backward(g_zr, zr, pre_r, h_r, norm_r, bag_r, right)
            w -= lr * (gw1 + gw2)
            b -= lr * (gb1 + gb2)
            for i, idx in enumerate(left):
                emb[idx] -= lr * gbag1[i] / len(idx)
            for i, idx in enumerate(right):
                emb[idx] -= lr * gbag2[i] / len(idx)
        losses.append(total / max(n, 1))
    return LabelEncoder(buckets, dim, emb, w, b), losses


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=None, help="captured paired frames (vision_data)")
    ap.add_argument("--cache", type=Path, default=None, help="cached recognizer output for those frames")
    ap.add_argument("--buckets", type=int, default=8192)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    pairs, counts = build_pairs(args.data, args.cache, args.seed)
    held = set()
    for label in HELD_OUT.values():
        held.add(normalize(label))
    leaked = [(a, c) for a, c in pairs if normalize(a) in held or normalize(c) in held]
    if leaked:
        raise SystemExit(f"held-out phrasing leaked into training data: {leaked[:3]}")

    t0 = time.perf_counter()
    encoder, losses = train(pairs, buckets=args.buckets, dim=args.dim, epochs=args.epochs, batch=args.batch, lr=args.lr, seed=args.seed)
    seconds = time.perf_counter() - t0
    encoder.save(args.out / "label_encoder.npz")

    params = args.buckets * args.dim + args.dim * args.dim + args.dim
    manifest = {
        "pairs": len(pairs), "sources": counts, "held_out": HELD_OUT, "epochs": args.epochs, "batch": args.batch, "lr": args.lr,
        "seed": args.seed, "buckets": args.buckets, "dim": args.dim, "parameters": params, "train_seconds": round(seconds, 1),
        "loss_first": round(losses[0], 4), "loss_last": round(losses[-1], 4),
        "data": str(args.data) if args.data else None,
    }
    (args.out / "label_encoder.manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
