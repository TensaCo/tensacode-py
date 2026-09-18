"""Train the extractive answerer on SQuAD 2.0 train, the split the evaluation never touches.

    python eval/training/train_span.py --epochs 2

This attacks the largest diagnosed bottleneck in docs/revival/13-schema-brittleness.md:
handed the gold sentence, the hand-written generator produced the gold span only 33.9% of
the time. Here the projection from evidence to answer is learned instead, including
"the passage does not answer this", which SQuAD 2.0 labels directly.
"""

from __future__ import annotations

import os

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from tensorcode.backends.neural import SpanAnswererModel  # noqa: E402

SP = Path(os.environ.get("TENSORCODE_SCRATCH", os.path.expanduser("~/.cache/tensorcode")))


class Squad(Dataset):
    """Question/context pairs with start and end token positions, CLS for unanswerable."""

    def __init__(self, rows: list[dict], tokenizer, max_length: int) -> None:
        self.rows, self.tok, self.max_length = rows, tokenizer, max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        row = self.rows[i]
        enc = self.tok(row["question"].strip(), row["context"], truncation="only_second",
                       max_length=self.max_length, return_offsets_mapping=True)
        offsets, seq = enc["offset_mapping"], enc.sequence_ids(0)
        start_pos = end_pos = 0  # CLS: no answer
        answerable = 0
        texts, starts = row["answers"]["text"], row["answers"]["answer_start"]
        if texts:
            a, b = starts[0], starts[0] + len(texts[0])
            first = next((k for k, s in enumerate(seq) if s == 1), None)
            last = next((k for k in range(len(seq) - 1, -1, -1) if seq[k] == 1), None)
            if first is not None and last is not None and offsets[first][0] <= a and offsets[last][1] >= b:
                s = next((k for k in range(first, last + 1) if offsets[k][0] <= a < offsets[k][1]), None)
                e = next((k for k in range(last, first - 1, -1) if offsets[k][0] < b <= offsets[k][1]), None)
                if s is not None and e is not None and s <= e:
                    start_pos, end_pos, answerable = s, e, 1
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
                "token_type_ids": enc.get("token_type_ids", [0] * len(enc["input_ids"])),
                "start": start_pos, "end": end_pos, "answerable": answerable}


def collate(items: list[dict], pad_id: int = 0) -> dict:
    """Pad to the longest in the batch, not to the model's maximum: most contexts are far shorter."""
    width = max(len(x["input_ids"]) for x in items)

    def pad(key: str, fill: int) -> torch.Tensor:
        return torch.tensor([x[key] + [fill] * (width - len(x[key])) for x in items])

    return {
        "input_ids": pad("input_ids", pad_id),
        "attention_mask": pad("attention_mask", 0),
        "token_type_ids": pad("token_type_ids", 0),
        "start": torch.tensor([x["start"] for x in items]),
        "end": torch.tensor([x["end"] for x in items]),
        "answerable": torch.tensor([x["answerable"] for x in items]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--encoder", default="google/electra-base-discriminator")
    ap.add_argument("--out", type=Path, default=SP / "artifacts" / "span-answerer")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max-length", type=int, default=384)
    ap.add_argument("--train-size", type=int, default=0, help="0 = all of SQuAD 2.0 train")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed), np.random.seed(args.seed), torch.manual_seed(args.seed)
    from datasets import load_dataset

    data = load_dataset("rajpurkar/squad_v2", split="train")
    rows = [dict(r) for r in data]
    random.Random(args.seed).shuffle(rows)
    if args.train_size:
        rows = rows[: args.train_size]
    print(f"SQuAD 2.0 train: {len(rows)} questions, {sum(1 for r in rows if not r['answers']['text'])} unanswerable", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.encoder)
    loader = DataLoader(Squad(rows, tok, args.max_length), batch_size=args.batch_size, shuffle=True,
                        collate_fn=lambda b: collate(b, tok.pad_token_id), num_workers=8, drop_last=True)
    model = SpanAnswererModel(args.encoder).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * len(loader)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=steps, pct_start=0.06)
    ce = torch.nn.CrossEntropyLoss()
    scaler_dtype = torch.bfloat16

    t0, history = time.perf_counter(), []
    model.train()
    for epoch in range(args.epochs):
        running = 0.0
        for step, b in enumerate(loader):
            with torch.autocast("cuda", dtype=scaler_dtype, enabled=device == "cuda"):
                got = model(b["input_ids"].to(device), b["attention_mask"].to(device), b["token_type_ids"].to(device))
                loss = ce(got["start"].float(), b["start"].to(device)) + ce(got["end"].float(), b["end"].to(device))
                loss = loss + ce(got["answerable"].float(), b["answerable"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); schedule.step(); optimizer.zero_grad(set_to_none=True)
            running += float(loss)
            if step % 200 == 0:
                print(f"epoch {epoch} step {step}/{len(loader)} loss {running / (step + 1):.4f} "
                      f"[{time.perf_counter() - t0:.0f}s]", flush=True)
        history.append({"epoch": epoch, "loss": running / len(loader)})

    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out / "weights.pt")
    tok.save_pretrained(str(args.out / "tokenizer"))
    config = {
        "version": "1", "encoder": args.encoder, "max_length": args.max_length,
        "parameters": sum(p.numel() for p in model.parameters()),
        "train_size": len(rows), "epochs": args.epochs, "seed": args.seed,
        "train_seconds": round(time.perf_counter() - t0, 1), "history": history,
        "threshold": 0.0,
        "measured_on": "trained on SQuAD 2.0 train; the evaluation uses the validation split only",
    }
    (args.out / "config.json").write_text(json.dumps(config, indent=1))
    print(json.dumps(config, indent=1))


if __name__ == "__main__":
    main()
