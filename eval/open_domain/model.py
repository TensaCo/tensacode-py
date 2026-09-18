"""The model arm: a local instruct model, prompted plainly, with no tensorcode structure.

Batched greedy decoding, so the comparison is not an artifact of throughput. Multiple
choice is scored by option likelihood (one forward pass per option), which is the
standard way to evaluate a model on ARC and avoids penalising formatting.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import Item

SYSTEMS = {
    "squad2": ("Answer the question using only the passage. Reply with the exact answer span and nothing else. "
               "If the passage does not contain the answer, reply exactly: unanswerable"),
    "hotpot": ("Answer the question using only the passages. Reply with the short answer (a name, date, number or "
               "yes/no) and nothing else."),
    "gsm8k": "Solve the problem. Think step by step, then end with a line of the form: #### <number>",
    "arc_easy": "Answer the multiple-choice question. Reply with the single letter of the correct option and nothing else.",
}
MAX_NEW = {"squad2": 24, "hotpot": 24, "gsm8k": 320, "arc_easy": 4}


@dataclass
class LocalModel:
    model_id: str
    device: str = "cuda"
    batch_size: int = 16
    calls: int = 0
    prompt_tokens: int = 0
    new_tokens: int = 0
    seconds: float = 0.0
    _tok: object = field(default=None, repr=False)
    _model: object = field(default=None, repr=False)

    def load(self) -> None:
        import time

        t0 = time.perf_counter()
        self._tok = AutoTokenizer.from_pretrained(self.model_id, padding_side="left")
        if self._tok.pad_token is None:
            self._tok.pad_token = self._tok.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(self.model_id, dtype=torch.bfloat16).to(self.device).eval()
        self.load_seconds = time.perf_counter() - t0

    def _chat(self, system: str, user: str) -> str:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            return self._tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return self._tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    @torch.inference_mode()
    def generate(self, benchmark: str, prompts: list[str]) -> list[str]:
        import time

        out: list[str] = []
        system = SYSTEMS[benchmark]
        for i in range(0, len(prompts), self.batch_size):
            chunk = [self._chat(system, p) for p in prompts[i : i + self.batch_size]]
            batch = self._tok(chunk, return_tensors="pt", padding=True).to(self.device)
            t0 = time.perf_counter()
            gen = self._model.generate(**batch, max_new_tokens=MAX_NEW[benchmark], do_sample=False,
                                       pad_token_id=self._tok.pad_token_id)
            self.seconds += time.perf_counter() - t0
            new = gen[:, batch["input_ids"].shape[1]:]
            self.calls += len(chunk)
            self.prompt_tokens += int(batch["input_ids"].numel())
            self.new_tokens += int((new != self._tok.pad_token_id).sum())
            out += [self._tok.decode(row, skip_special_tokens=True).strip() for row in new]
        return out

    @torch.inference_mode()
    def score_options(self, item: Item) -> str:
        """Pick the option with the highest mean log-likelihood given the question."""
        import time

        prefix = f"Question: {item.question}\nAnswer:"
        best, best_lp = None, -1e9
        t0 = time.perf_counter()
        for label, text in item.options.items():
            ids = self._tok(prefix, return_tensors="pt").input_ids.to(self.device)
            full = self._tok(prefix + " " + text, return_tensors="pt").input_ids.to(self.device)
            logits = self._model(full).logits[0, :-1]
            targets = full[0, 1:]
            lp = torch.log_softmax(logits.float(), dim=-1).gather(1, targets[:, None])[ids.shape[1] - 1 :]
            mean = float(lp.mean())
            if mean > best_lp:
                best, best_lp = label, mean
        self.seconds += time.perf_counter() - t0
        self.calls += 1
        return best or sorted(item.options)[0]


def prompt_for(benchmark: str, item: Item) -> str:
    if benchmark == "squad2":
        ctx = " ".join(s for _, s in item.passages)
        return f"Passage: {ctx}\n\nQuestion: {item.question}"
    if benchmark == "hotpot":
        by_title: dict[str, list[str]] = {}
        for t, s in item.passages:
            by_title.setdefault(t, []).append(s)
        ctx = "\n\n".join(f"{t}: {' '.join(ss)}" for t, ss in by_title.items())
        return f"{ctx}\n\nQuestion: {item.question}"
    if benchmark == "gsm8k":
        return item.question
    if benchmark == "arc_easy":
        opts = "\n".join(f"{k}. {v}" for k, v in sorted(item.options.items()))
        return f"{item.question}\n{opts}"
    raise ValueError(benchmark)


FINAL = re.compile(r"####\s*(-?[\d,.]+)")


def parse_reply(benchmark: str, text: str) -> str:
    t = text.strip()
    if benchmark == "gsm8k":
        m = FINAL.search(t)
        if m:
            return m.group(1)
        nums = re.findall(r"-?\d[\d,]*\.?\d*", t)
        return nums[-1] if nums else t[:40]
    if benchmark == "arc_easy":
        m = re.search(r"\b([A-D1-4])\b", t)
        return m.group(1) if m else t[:1]
    first = t.splitlines()[0].strip() if t else ""
    return first.strip('"').strip()
