"""Learn which intention to take, from episodes the apps themselves graded.

    PYTHONPATH=src:. python eval/training/train_intention_ranker.py --train FILE --held-out FILE --out DIR

Every decision in the rollout file lists the options the agent faced, which one the
hand-written objective took, and whether that episode ended fully correct according to
the app's own scoring. That is a behaviour-cloning set with a reward: imitate the choice,
weighted by whether the episode it belonged to worked.

``priority`` is withheld on purpose. It *is* the hand-written objective, so a ranker given
it would score 100% by copying one number and would have learned nothing. The features are
the kind of intention, what it says it is doing (through the learned label encoder), and
the shape of the option set.

The comparison that matters is not held-out imitation accuracy — it is whether an agent
choosing this way still finishes its tasks. That is measured separately, by running
episodes with this ranker in place of the objective (``eval_intention_ranker.py``).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).parents[2] / "src"), str(Path(__file__).parents[2])]

from examples.browser_agents.learned.text_embed import LabelEncoder  # noqa: E402

KINDS = ("Press", "Enter", "Note", "Wait", "Finish", "Escalate", "Type", "Start", "Advance", "Drop")
NUMERIC = ("records", "submits", "text_len", "has_text", "claims", "wait_ms", "n_options", "index", "why_len", "has_control")


#: The option lists the agents build are already ordered by the hand-written code that
#: generated them, so a candidate's position leaks the answer: on held-out decisions
#: "always take the first option" scored exactly what the trained ranker scored (0.7992).
#: With POSITIONAL False the index feature is dropped and options are shuffled, so the
#: ranker has to decide from what an intention is rather than where it sits.
POSITIONAL = False


def featurize(candidate: dict, *, index: int, n: int, encoder: LabelEncoder | None) -> np.ndarray:
    kind = [1.0 if candidate["kind"] == k else 0.0 for k in KINDS]
    numeric = [
        candidate.get("records", 0), float(candidate.get("submits", False)), min(candidate.get("text_len", 0), 80) / 80,
        float(candidate.get("has_text", False)), candidate.get("claims", 0), min(candidate.get("wait_ms", 0.0), 100) / 100,
        n / 10, (index / 10) if POSITIONAL else 0.0, min(len(candidate.get("why", "")), 160) / 160, float(bool(candidate.get("control"))),
    ]
    text = encoder.encode([candidate.get("why", "")])[0] if encoder is not None else np.zeros(0, dtype=np.float32)
    return np.concatenate([np.array(kind, dtype=np.float32), np.array(numeric, dtype=np.float32), text])


def load(path: Path, encoder: LabelEncoder | None, *, shuffle: bool = True, seed: int = 0) -> list[dict]:
    rows = []
    rng = np.random.default_rng(seed)
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row.get("chosen") is None or row["n"] < 2:
            continue
        candidates, chosen = row["candidates"], row["chosen"]
        if shuffle:  # break the ordering the generating code imposed
            order = rng.permutation(len(candidates))
            candidates = [candidates[i] for i in order]
            chosen = int(np.where(order == chosen)[0][0])
        row["candidates"], row["chosen"] = candidates, chosen
        row["x"] = np.stack([featurize(c, index=i, n=row["n"], encoder=encoder) for i, c in enumerate(candidates)])
        rows.append(row)
    return rows


def train(rows: list[dict], *, dim: int, epochs: int, lr: float, seed: int, use_reward: bool) -> tuple[np.ndarray, list[float]]:
    """A linear scorer trained with a softmax-over-options loss (Plackett-Luce, top-1)."""
    rng = np.random.default_rng(seed)
    w = np.zeros(dim, dtype=np.float32)
    losses = []
    order = np.arange(len(rows))
    for _ in range(epochs):
        rng.shuffle(order)
        total = 0.0
        for i in order:
            row = rows[i]
            weight = 1.0 if not use_reward else (1.0 if row["episode_ok"] else 0.2)
            scores = row["x"] @ w
            m = scores.max()
            p = np.exp(scores - m)
            p /= p.sum()
            target = np.zeros(len(scores), dtype=np.float32)
            target[row["chosen"]] = 1.0
            total += -float(np.log(max(p[row["chosen"]], 1e-12))) * weight
            w -= lr * weight * ((p - target) @ row["x"])
        losses.append(total / max(len(rows), 1))
    return w, losses


def top1(rows: list[dict], w: np.ndarray) -> float:
    if not rows:
        return float("nan")
    hits = sum(int(np.argmax(row["x"] @ w) == row["chosen"]) for row in rows)
    return hits / len(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--held-out", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--encoder", type=Path, default=None)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    encoder = LabelEncoder.load(args.encoder) if args.encoder else None
    train_rows = load(args.train, encoder, shuffle=not POSITIONAL, seed=args.seed)
    held_rows = load(args.held_out, encoder, shuffle=not POSITIONAL, seed=args.seed + 7)
    if not train_rows:
        raise SystemExit("no usable decisions in the training file")
    dim = train_rows[0]["x"].shape[1]

    t0 = time.perf_counter()
    w, losses = train(train_rows, dim=dim, epochs=args.epochs, lr=args.lr, seed=args.seed, use_reward=True)
    seconds = time.perf_counter() - t0
    w_noreward, _ = train(train_rows, dim=dim, epochs=args.epochs, lr=args.lr, seed=args.seed, use_reward=False)

    # controls: a scorer that cannot work, and the simplest rule that can
    rng = np.random.default_rng(args.seed + 1)
    w_random = rng.standard_normal(dim).astype(np.float32)
    first_option = sum(int(row["chosen"] == 0) for row in held_rows) / max(len(held_rows), 1)

    per_task = {}
    for task in sorted({row["task"] for row in held_rows}):
        sel = [r for r in held_rows if r["task"] == task]
        per_task[task] = {"n": len(sel), "learned": round(top1(sel, w), 4), "random": round(top1(sel, w_random), 4),
                          "always_first": round(sum(int(r["chosen"] == 0) for r in sel) / len(sel), 4)}

    report = {
        "train_file": str(args.train), "held_out_file": str(args.held_out),
        "train_decisions": len(train_rows), "held_out_decisions": len(held_rows),
        "features": {"kinds": list(KINDS), "numeric": list(NUMERIC), "text_dim": dim - len(KINDS) - len(NUMERIC), "withheld": ["priority"] + ([] if POSITIONAL else ["index (options shuffled)"])},
        "options_shuffled": not POSITIONAL,
        "parameters": dim, "epochs": args.epochs, "lr": args.lr, "seed": args.seed,
        "train_seconds": round(seconds, 2), "loss_first": round(losses[0], 4), "loss_last": round(losses[-1], 4),
        "imitation_top1": {
            "train": round(top1(train_rows, w), 4),
            "held_out": round(top1(held_rows, w), 4),
            "held_out_no_reward_weighting": round(top1(held_rows, w_noreward), 4),
            "held_out_random_control": round(top1(held_rows, w_random), 4),
            "held_out_always_first_option": round(first_option, 4),
        },
        "per_task_held_out": per_task,
        "option_set_sizes": dict(Counter(r["n"] for r in train_rows)),
        "kind_distribution": dict(Counter(c["kind"] for r in train_rows for c in r["candidates"])),
    }
    np.savez(args.out / "intention_ranker.npz", w=w, meta=np.array([dim]))
    (args.out / "intention_ranker.report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ("train_decisions", "held_out_decisions", "parameters", "train_seconds", "imitation_top1", "per_task_held_out")}, indent=1))


if __name__ == "__main__":
    main()
