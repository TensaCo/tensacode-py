"""Learn claim embeddings from what the agents thought together, for cue-based recall.

    PYTHONPATH=src:. python eval/training/train_claim_embeddings.py --recording FILE --out DIR

The awareness core wants to spread activation from a cue to related claims. Hand-built
links (shared subject, shared object, provenance) are exact but narrow: they never connect
``ticket:T-100 department Finance`` to ``ui:...Department#1 may_mean department`` unless
somebody wrote that edge. The recordings contain the association for free — claims that
arrived in the same cycle of the same episode were relevant to each other at that moment.

Trained with skip-gram and negative sampling over those co-occurrences (numpy, fixed
seed). Held-out episodes are never trained on, and the baseline is token overlap, which is
what a string-matching recall would give.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).parents[2] / "src"), str(Path(__file__).parents[2])]

_TOKEN = re.compile(r"[A-Za-z0-9_:./@-]+")


def tokens(claim: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(claim)][:12]


def subject_of(claim: str) -> str:
    parts = claim.split(" ", 1)
    return parts[0] if parts else claim


def read_cycles(path: Path, *, limit: int) -> list[dict]:
    """(episode, cycle) -> the claim lines the agent newly believed, from a live recording."""
    cycles = []
    with path.open() as fh:
        for line in fh:
            if '"type": "cycle"' not in line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            thoughts = [t for t in (ev.get("thoughts") or []) if t and " " in t]
            if len(thoughts) < 2:
                continue
            cycles.append({"agent": ev.get("agent"), "cycle": ev.get("cycle"), "claims": thoughts})
            if len(cycles) >= limit:
                break
    return cycles


def build_vocab(cycles: list[dict], min_count: int) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for c in cycles:
        for claim in c["claims"]:
            counts.update(set(tokens(claim)))
    return {t: i for i, (t, n) in enumerate(counts.most_common()) if n >= min_count}


def embed_claim(claim: str, vocab: dict[str, int], emb: np.ndarray) -> np.ndarray:
    idx = [vocab[t] for t in tokens(claim) if t in vocab]
    if not idx:
        return np.zeros(emb.shape[1], dtype=np.float32)
    v = emb[idx].mean(axis=0)
    return v / max(float(np.linalg.norm(v)), 1e-8)


def train(cycles: list[dict], vocab: dict[str, int], *, dim: int, epochs: int, lr: float, negatives: int, seed: int) -> tuple[np.ndarray, list[float]]:
    rng = np.random.default_rng(seed)
    emb = (rng.standard_normal((len(vocab), dim)) * 0.05).astype(np.float32)
    ctx = (rng.standard_normal((len(vocab), dim)) * 0.05).astype(np.float32)
    freq = np.ones(len(vocab), dtype=np.float64)
    for c in cycles:
        for claim in c["claims"]:
            for t in tokens(claim):
                if t in vocab:
                    freq[vocab[t]] += 1
    noise = freq ** 0.75
    noise /= noise.sum()
    pairs = []
    for c in cycles:
        ids = [[vocab[t] for t in tokens(claim) if t in vocab] for claim in c["claims"]]
        for i, a in enumerate(ids):
            for b in ids[i + 1 :]:
                for u in a:
                    for v in b:
                        pairs.append((u, v))
    pairs_arr = np.array(pairs, dtype=np.int32)
    losses = []
    for _ in range(epochs):
        rng.shuffle(pairs_arr)
        total = 0.0
        for start in range(0, len(pairs_arr), 4096):
            chunk = pairs_arr[start : start + 4096]
            u, v = chunk[:, 0], chunk[:, 1]
            neg = rng.choice(len(vocab), size=(len(chunk), negatives), p=noise)
            eu, cv = emb[u], ctx[v]
            pos = np.clip(np.sum(eu * cv, axis=1), -30, 30)  # keep the sigmoid out of overflow
            sig_pos = 1 / (1 + np.exp(-pos))
            gp = (sig_pos - 1.0)[:, None]
            cn = ctx[neg]
            negs = np.clip(np.einsum("bd,bkd->bk", eu, cn), -30, 30)
            sig_neg = 1 / (1 + np.exp(-negs))
            total += float(-np.log(np.maximum(sig_pos, 1e-9)).mean() - np.log(np.maximum(1 - sig_neg, 1e-9)).mean())
            g_eu = gp * cv + np.einsum("bk,bkd->bd", sig_neg, cn)
            np.add.at(ctx, v, -lr * gp * eu)
            np.add.at(ctx, neg.ravel(), -lr * (sig_neg.ravel()[:, None] * np.repeat(eu, negatives, axis=0)))
            np.add.at(emb, u, -lr * g_eu)
            # np.add.at sums duplicate indices, so a token appearing many times in one
            # batch takes a huge step; keep every row on the unit ball instead
            for table in (emb, ctx):
                norms = np.linalg.norm(table, axis=1, keepdims=True)
                np.divide(table, np.maximum(norms, 1.0), out=table)
        losses.append(total / max(1, len(pairs_arr) // 4096))
    if not np.isfinite(emb).all():
        raise SystemExit(f"training diverged: {int(np.isnan(emb).any(axis=1).sum())} of {len(emb)} rows are not finite")
    return emb, losses


def train_ppmi_svd(cycles: list[dict], vocab: dict[str, int], *, dim: int, seed: int) -> tuple[np.ndarray, dict]:
    """Positive pointwise mutual information over co-occurrence, factorised by SVD.

    A closed-form alternative to the sampled objective above: no learning rate, no
    divergence, deterministic for a fixed matrix. Co-occurrence is counted between tokens
    of claims that arrived in the same cycle, which is the association we want.
    """
    n = len(vocab)
    counts = np.zeros((n, n), dtype=np.float64)
    for c in cycles:
        ids = [[vocab[t] for t in tokens(claim) if t in vocab] for claim in c["claims"]]
        for i, a in enumerate(ids):
            for b in ids[i + 1 :]:
                for u in a:
                    for v in b:
                        counts[u, v] += 1
                        counts[v, u] += 1
    total = counts.sum()
    if total == 0:
        raise SystemExit("no co-occurrences found")
    row = counts.sum(axis=1, keepdims=True)
    col = counts.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log((counts * total) / np.maximum(row * col, 1e-12))
    ppmi = np.nan_to_num(np.maximum(pmi, 0.0), nan=0.0, posinf=0.0, neginf=0.0)
    u, sv, _ = np.linalg.svd(ppmi, full_matrices=False)
    emb = (u[:, :dim] * np.sqrt(sv[:dim])).astype(np.float32)
    if not np.isfinite(emb).all():
        raise SystemExit("PPMI/SVD produced non-finite embeddings")
    return emb, {"method": "ppmi-svd", "nonzero_ppmi": int((ppmi > 0).sum()), "top_singular_values": [round(float(x), 3) for x in sv[:5]]}


def overlap(a: str, b: str) -> float:
    ta, tb = set(tokens(a)), set(tokens(b))
    return len(ta & tb) / max(len(ta | tb), 1)


def evaluate(cycles: list[dict], vocab: dict[str, int], emb: np.ndarray, *, k: int = 5, sample: int = 300, seed: int = 0) -> dict:
    """Given one claim as a cue, do the top-k recalled claims belong with it?

    Two notions of 'belong': they were believed in the same cycle (associative), or they
    describe the same subject (structural). The pool is every claim in the split, so this
    is retrieval, not a binary judgement.
    """
    rng = np.random.default_rng(seed)
    pool, cycle_of, subj = [], [], []
    for i, c in enumerate(cycles):
        for claim in c["claims"]:
            pool.append(claim)
            cycle_of.append(i)
            subj.append(subject_of(claim))
    vecs = np.stack([embed_claim(p, vocab, emb) for p in pool])
    idxs = rng.choice(len(pool), size=min(sample, len(pool)), replace=False)
    out = {"n_pool": len(pool), "n_cues": len(idxs)}
    for arm in ("learned", "token_overlap"):
        same_cycle, same_subject = [], []
        for i in idxs:
            if arm == "learned":
                scores = vecs @ vecs[i]
            else:
                scores = np.array([overlap(pool[i], p) for p in pool])
            scores[i] = -np.inf
            top = np.argsort(-scores)[:k]
            same_cycle.append(float(np.mean([cycle_of[j] == cycle_of[i] for j in top])))
            same_subject.append(float(np.mean([subj[j] == subj[i] for j in top])))
        out[arm] = {f"same_cycle_p@{k}": round(float(np.mean(same_cycle)), 4), f"same_subject_p@{k}": round(float(np.mean(same_subject)), 4)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recording", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cycles", type=int, default=6000)
    ap.add_argument("--dim", type=int, default=48)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--negatives", type=int, default=5)
    ap.add_argument("--min-count", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--method", choices=("ppmi-svd", "sgns"), default="ppmi-svd")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cycles = read_cycles(args.recording, limit=args.cycles)
    if len(cycles) < 50:
        raise SystemExit(f"only {len(cycles)} usable cycles in {args.recording}")
    split = int(0.8 * len(cycles))
    train_cycles, held_cycles = cycles[:split], cycles[split:]
    vocab = build_vocab(train_cycles, args.min_count)
    t0 = time.perf_counter()
    if args.method == "ppmi-svd":
        emb, extra = train_ppmi_svd(train_cycles, vocab, dim=args.dim, seed=args.seed)
        losses = [float("nan")]
    else:
        emb, losses = train(train_cycles, vocab, dim=args.dim, epochs=args.epochs, lr=args.lr, negatives=args.negatives, seed=args.seed)
        extra = {"method": "sgns"}
    seconds = time.perf_counter() - t0

    report = {
        "recording": str(args.recording), "cycles_read": len(cycles), "train_cycles": len(train_cycles), "held_out_cycles": len(held_cycles),
        "vocab": len(vocab), "dim": args.dim, "parameters": len(vocab) * args.dim, "epochs": args.epochs, "seed": args.seed,
        "train_seconds": round(seconds, 1), "loss_first": round(losses[0], 4), "loss_last": round(losses[-1], 4), **extra,
        "held_out": evaluate(held_cycles, vocab, emb, seed=args.seed),
        "train_split": evaluate(train_cycles, vocab, emb, seed=args.seed, sample=200),
    }
    np.savez(args.out / "claim_embeddings.npz", emb=emb, vocab=np.array(list(vocab), dtype=object), dim=np.array([args.dim]))
    (args.out / "claim_embeddings.report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
