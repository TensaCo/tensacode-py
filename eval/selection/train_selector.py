"""Train a cross-encoder that scores one sentence against one question.

    PYTHONPATH=src:. venv-eval/bin/python eval/selection/train_selector.py --items 12000

Doc 21 bracketed the multi-hop gap: a perfect selector over a realistic pool is worth +0.087 EM,
and three lexical criteria (BM25 relevance, entity-linked pairs, question coverage) all landed on
the same curve, unable to reach it. The labels needed to do it properly are free and mechanically
true — HotpotQA's train split marks which sentences support each answer — so this is a capability
gap, not a schema gap, and the honest response is to train the thing.

One question and one sentence go in together (a cross-encoder, not two towers), because whether a
sentence matters depends on the question's own terms. Negatives are the other ~38 sentences of the
same item, which is the distribution the selector will actually face: distractor paragraphs about
entities the question mentions.
"""

from __future__ import annotations

import os

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

SP = Path(os.environ.get("TENSACODE_SCRATCH", os.path.expanduser("~/.cache/tensacode")))


def pairs(examples, *, negatives_per_gold: int, rng: random.Random) -> list[tuple[str, str, int]]:
    """(question, sentence, label). Negatives are sampled from the item's own distractors."""
    out = []
    for ex in examples:
        gold = [s for s in ex.sentences if s.gold]
        rest = [s for s in ex.sentences if not s.gold]
        rng.shuffle(rest)
        for s in gold:
            out.append((ex.question, f"{s.title}. {s.text}", 1))
        for s in rest[: negatives_per_gold * max(1, len(gold))]:
            out.append((ex.question, f"{s.title}. {s.text}", 0))
    rng.shuffle(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--items", type=int, default=12000, help="training questions")
    ap.add_argument("--negatives", type=int, default=4, help="negatives per gold sentence")
    ap.add_argument("--encoder", default="google/electra-small-discriminator")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max-length", type=int, default=160)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=SP / "artifacts" / "sentence-selector")
    args = ap.parse_args()

    import torch
    from torch import nn
    from transformers import AutoModel, AutoTokenizer

    from eval.selection.data import train as load_train

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    t0 = time.time()
    examples = load_train(args.items, seed=args.seed)
    rows = pairs(examples, negatives_per_gold=args.negatives, rng=rng)
    print(f"{len(examples)} questions -> {len(rows)} pairs "
          f"({sum(l for _, _, l in rows)} positive), {time.time() - t0:.0f}s", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.encoder)
    encoder = AutoModel.from_pretrained(args.encoder)
    head = nn.Linear(encoder.config.hidden_size, 1)

    class Selector(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder, self.head = encoder, head

        def forward(self, **enc):
            return self.head(self.encoder(**enc).last_hidden_state[:, 0]).squeeze(-1)

    model = Selector().to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    steps = (len(rows) // args.batch_size) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=max(1, steps), pct_start=0.1)
    loss_fn = nn.BCEWithLogitsLoss()

    history = []
    t0 = time.time()
    for epoch in range(args.epochs):
        rng.shuffle(rows)
        total, seen = 0.0, 0
        for i in range(0, len(rows) - args.batch_size + 1, args.batch_size):
            batch = rows[i : i + args.batch_size]
            enc = tokenizer([q for q, _, _ in batch], [s for _, s, _ in batch], return_tensors="pt",
                            padding=True, truncation=True, max_length=args.max_length).to(device)
            y = torch.tensor([float(l) for _, _, l in batch], device=device)
            loss = loss_fn(model(**enc), y)
            loss.backward()
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            total += float(loss) * len(batch)
            seen += len(batch)
            if seen % (args.batch_size * 100) == 0:
                print(f"  epoch {epoch} {seen}/{len(rows)} loss {total / seen:.4f} "
                      f"{time.time() - t0:.0f}s", flush=True)
        history.append({"epoch": epoch, "loss": round(total / max(1, seen), 5)})

    args.out.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, args.out / "weights.pt")
    tokenizer.save_pretrained(args.out / "tokenizer")
    (args.out / "config.json").write_text(json.dumps({
        "version": "1", "encoder": args.encoder, "max_length": args.max_length,
        "parameters": sum(p.numel() for p in model.parameters()),
        "train_questions": len(examples), "train_pairs": len(rows), "negatives_per_gold": args.negatives,
        "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr, "seed": args.seed,
        "train_seconds": round(time.time() - t0, 1), "history": history,
        "data": "HotpotQA distractor train (official split), labels = dataset supporting facts",
        "measured_on": "evaluated on HotpotQA distractor validation only; train/eval disjoint by split",
    }, indent=1))
    print(f"saved {args.out} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
