"""Train the learned request parser: utterance -> (act, slots).

    python eval/training/train_parser.py --epochs 3

Fixed seed, one held-out dev slice from the generated pool, and the artifact records its
own label space so inference cannot drift from training. The evaluation sets named in the
data manifest are never seen here.
"""

from __future__ import annotations

import os

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from eval.training import schema as S  # noqa: E402
from eval.training.parser_data import Example, read  # noqa: E402
from tensorcode.backends.neural import RequestParserModel  # noqa: E402

SP = Path(os.environ.get("TENSORCODE_SCRATCH", os.path.expanduser("~/.cache/tensorcode")))


@dataclass
class Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    act: torch.Tensor
    tags: torch.Tensor
    closed: torch.Tensor
    flags: torch.Tensor


class Utterances(Dataset):
    def __init__(self, rows: list[Example], tokenizer, max_length: int) -> None:
        self.rows, self.tok, self.max_length = rows, tokenizer, max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        ex = self.rows[i]
        enc = self.tok(ex.text, truncation=True, max_length=self.max_length, return_offsets_mapping=True)
        offsets = enc["offset_mapping"]
        tags = [S.TAG_INDEX["O"]] * len(offsets)
        for slot, (start, end) in ex.spans.items():
            if slot not in S.SPAN_INDEX:
                continue
            first = True
            for position, (a, b) in enumerate(offsets):
                if a == b:  # special token
                    continue
                if a >= start and b <= end:
                    tags[position] = S.TAG_INDEX[f"{'B' if first else 'I'}-{slot}"]
                    first = False
        for position, (a, b) in enumerate(offsets):
            if a == b:
                tags[position] = -100  # ignored by the loss
        closed = [vocab.index(ex.closed.get(name, "none")) if ex.closed.get(name, "none") in vocab else 0
                  for name, vocab in S.CLOSED_HEADS]
        flags = [float(bool(ex.flags.get(f, False))) for f in S.FLAGS]
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "tags": tags,
                "act": S.ACT_INDEX.get(ex.act, 0), "closed": closed, "flags": flags}


def collate(items: list[dict], pad_id: int) -> Batch:
    width = max(len(x["input_ids"]) for x in items)
    ids = torch.full((len(items), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(items), width), dtype=torch.long)
    tags = torch.full((len(items), width), -100, dtype=torch.long)
    for i, x in enumerate(items):
        n = len(x["input_ids"])
        ids[i, :n] = torch.tensor(x["input_ids"])
        mask[i, :n] = torch.tensor(x["attention_mask"])
        tags[i, :n] = torch.tensor(x["tags"])
    return Batch(ids, mask, torch.tensor([x["act"] for x in items]), tags,
                 torch.tensor([x["closed"] for x in items]), torch.tensor([x["flags"] for x in items]))


def evaluate(model: RequestParserModel, loader: DataLoader, device: str) -> dict:
    model.eval()
    act_ok = act_n = tag_ok = tag_n = closed_ok = closed_n = flag_ok = flag_n = 0
    with torch.inference_mode():
        for b in loader:
            got = model(b.input_ids.to(device), b.attention_mask.to(device))
            act_ok += int((got["act"].argmax(-1).cpu() == b.act).sum()); act_n += len(b.act)
            keep = b.tags != -100
            tag_ok += int((got["tags"].argmax(-1).cpu()[keep] == b.tags[keep]).sum()); tag_n += int(keep.sum())
            for k, head in enumerate(got["closed"]):
                closed_ok += int((head.argmax(-1).cpu() == b.closed[:, k]).sum()); closed_n += len(b.act)
            flag_ok += int(((got["flags"] > 0).float().cpu() == b.flags).sum()); flag_n += b.flags.numel()
    model.train()
    return {"act_accuracy": act_ok / max(1, act_n), "tag_accuracy": tag_ok / max(1, tag_n),
            "closed_accuracy": closed_ok / max(1, closed_n), "flag_accuracy": flag_ok / max(1, flag_n)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=SP / "training")
    ap.add_argument("--out", type=Path, default=SP / "artifacts" / "request-parser")
    ap.add_argument("--encoder", default="google/electra-small-discriminator")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max-length", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0,
                    help="dataloader workers; 0 keeps tokenisation in-process. Worker processes "
                         "deadlocked here after a killed run left semaphores in /dev/shm, and "
                         "tokenising 48 tokens is not the bottleneck anyway.")
    args = ap.parse_args()

    random.seed(args.seed), np.random.seed(args.seed), torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.encoder)
    train_rows, dev_rows = read(args.data / "parser_train.jsonl"), read(args.data / "parser_dev.jsonl")
    train = DataLoader(Utterances(train_rows, tok, args.max_length), batch_size=args.batch_size, shuffle=True,
                       collate_fn=lambda b: collate(b, tok.pad_token_id), num_workers=args.workers, drop_last=True)
    dev = DataLoader(Utterances(dev_rows, tok, args.max_length), batch_size=args.batch_size,
                     collate_fn=lambda b: collate(b, tok.pad_token_id))

    model = RequestParserModel(args.encoder, len(S.ACTS), len(S.TAGS), [len(v) for _, v in S.CLOSED_HEADS], len(S.FLAGS)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * len(train)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=steps, pct_start=0.1)
    ce = torch.nn.CrossEntropyLoss()
    ce_tags = torch.nn.CrossEntropyLoss(ignore_index=-100)
    bce = torch.nn.BCEWithLogitsLoss()

    t0, history = time.perf_counter(), []
    for epoch in range(args.epochs):
        running = 0.0
        for step, b in enumerate(train):
            got = model(b.input_ids.to(device), b.attention_mask.to(device))
            loss = ce(got["act"], b.act.to(device))
            loss = loss + ce_tags(got["tags"].reshape(-1, len(S.TAGS)), b.tags.reshape(-1).to(device))
            for k, head in enumerate(got["closed"]):
                loss = loss + 0.5 * ce(head, b.closed[:, k].to(device))
            loss = loss + 0.5 * bce(got["flags"], b.flags.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); schedule.step(); optimizer.zero_grad(set_to_none=True)
            running += float(loss)
            if step % 100 == 0:
                print(f"epoch {epoch} step {step}/{len(train)} loss {running / (step + 1):.4f}", flush=True)
        scores = evaluate(model, dev, device)
        history.append({"epoch": epoch, "loss": running / len(train), **scores})
        print(json.dumps(history[-1]), flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out / "weights.pt")
    tok.save_pretrained(str(args.out / "tokenizer"))
    params = sum(p.numel() for p in model.parameters())
    config = {
        "version": "1",
        "encoder": args.encoder,
        "max_length": args.max_length,
        "acts": list(S.ACTS),
        "tags": list(S.TAGS),
        "closed_heads": [[name, list(vocab)] for name, vocab in S.CLOSED_HEADS],
        "flags": list(S.FLAGS),
        "act_slots": {a: sorted(S.slots_of(a)) for a in S.ACTS},
        "whole_input_slots": sorted(S.WHOLE_INPUT_SLOTS),
        "threshold": 0.0,  # set by eval/training/calibrate_parser.py on a held-out slice
        "parameters": params,
        "train_size": len(train_rows),
        "dev_size": len(dev_rows),
        "epochs": args.epochs,
        "seed": args.seed,
        "train_seconds": round(time.perf_counter() - t0, 1),
        "history": history,
        "measured_on": "declared: trained on generated data (eval/training/parser_data.py); dev slice of the same pool",
        "quality": {"dev_act_accuracy": round(history[-1]["act_accuracy"], 4)} if history else {},
    }
    (args.out / "config.json").write_text(json.dumps(config, indent=1))
    print(json.dumps({k: v for k, v in config.items() if k not in ("acts", "tags", "closed_heads", "act_slots")}, indent=1))


if __name__ == "__main__":
    main()
