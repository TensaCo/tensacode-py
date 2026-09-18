"""Have a local model rewrite template utterances, keeping the label.

    python eval/training/paraphrase.py --n 4000

The rewrite is kept only if every span value still appears verbatim, so the carried-over
label is exact rather than assumed; everything else is dropped. Output is split into a
training half and an evaluation half from disjoint template seeds, and both are marked
``model-written`` in the manifest — a model wrote the surface form, so this measures
generalisation to phrasing neither the grammar nor the templates contain, not to language
a person actually typed.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from eval.training.parser_data import Example, generate_templates  # noqa: E402

SP = Path("/tmp/claude-1000/-home-brandonin-Documents-tensacode-tensacode-python/c572c14b-5662-4c07-8a7a-1ba7821d2bfa/scratchpad")

SYSTEM = """You rewrite one instruction a person typed to a computer assistant, in a different way a real person might type it.
Rules: keep every name, filename, path, quoted phrase and number EXACTLY as written, character for character.
Change the wording around them: word order, politeness, contractions, abbreviations, a different verb with the same meaning.
Keep it one line. Reply with the rewritten line only, nothing else."""


def rewrite(model, tokenizer, texts: list[str], *, batch_size: int, max_new_tokens: int = 48) -> list[str]:
    import torch

    out: list[str] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "system", "content": SYSTEM}, {"role": "user", "content": t}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            for t in chunk
        ]
        batch = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.inference_mode():
            got = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        for row, new in zip(chunk, got):
            text = tokenizer.decode(new[batch["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            out.append(text.splitlines()[0].strip().strip('"') if text else "")
        print(f"  rewrote {min(i + batch_size, len(texts))}/{len(texts)}", flush=True)
    return out


def carry_label(original: Example, text: str) -> Example | None:
    """Keep the rewrite only when every span value survives verbatim, so the label stays exact."""
    if not text or len(text) > 200:
        return None
    spans: dict[str, list[int]] = {}
    for slot, (a, b) in original.spans.items():
        value = original.text[a:b]
        at = text.find(value)
        if at < 0:  # the value was changed: the carried label would be a lie
            return None
        spans[slot] = [at, at + len(value)]
    if text.strip().lower() == original.text.strip().lower():
        return None  # not a rewrite
    return Example(text, original.act, spans, original.closed, original.flags, source="paraphrase")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--eval-fraction", type=float, default=0.25)
    ap.add_argument("--out", type=Path, default=SP / "training")
    ap.add_argument("--seed", type=int, default=101)  # disjoint from the training generator's seed
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    seeds = generate_templates(args.n, seed=args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to("cuda").eval()
    print(f"loaded {args.model} in {time.perf_counter() - t0:.0f}s", flush=True)

    t0 = time.perf_counter()
    rewrites = rewrite(model, tokenizer, [e.text for e in seeds], batch_size=args.batch_size)
    seconds = time.perf_counter() - t0
    kept = [k for k in (carry_label(e, r) for e, r in zip(seeds, rewrites)) if k is not None]

    rng = random.Random(args.seed)
    rng.shuffle(kept)
    cut = int(len(kept) * args.eval_fraction)
    hold, train = kept[:cut], kept[cut:]
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "paraphrase_train.jsonl").open("w") as fh:
        for ex in train:
            fh.write(json.dumps(asdict(ex)) + "\n")
    with (args.out / "paraphrase_eval.jsonl").open("w") as fh:
        for ex in hold:
            fh.write(json.dumps(asdict(ex)) + "\n")
    manifest = {
        "model": args.model, "asked": len(seeds), "kept": len(kept),
        "kept_fraction": round(len(kept) / max(1, len(seeds)), 4),
        "dropped_because_a_value_changed": len(seeds) - len(kept),
        "train": len(train), "eval": len(hold),
        "generation_seconds": round(seconds, 1),
        "seed": args.seed,
        "provenance": "surface forms written by a local model; labels carried from the template and value-checked",
    }
    (args.out / "paraphrase_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))
    for ex in hold[:8]:
        print(f"  {ex.act:<14} {ex.text}")


if __name__ == "__main__":
    main()
