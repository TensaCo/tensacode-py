"""A general instruction-tuned model running locally through ``transformers``.

Requires the ``local-model`` extra (torch, transformers). Greedy decoding, so
outputs are deterministic for a fixed model, prompt version, and software stack.
Outputs that are not exactly one allowed label are treated as abstentions,
never coerced.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..outcomes import Unknown
from ..runtime import Output, Profile, Request, Traits


class LocalChatModel:
    def __init__(self, model_id: str, *, device: str = "cuda", dtype: torch.dtype = torch.bfloat16, template_kwargs: dict | None = None) -> None:
        self.model_id = model_id
        self.template_kwargs = template_kwargs or {}
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device).eval()
        self.device = device

    @torch.inference_mode()
    def complete(self, prompts: Sequence[tuple[str, str]], *, max_new_tokens: int = 16) -> list[str]:
        texts = [
            self.tokenizer.apply_chat_template(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                tokenize=False,
                add_generation_prompt=True,
                **self.template_kwargs,
            )
            for system, user in prompts
        ]
        batch = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.device)
        out = self.model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=self.tokenizer.pad_token_id)
        return [self.tokenizer.decode(row[batch["input_ids"].shape[1] :], skip_special_tokens=True).strip() for row in out]


PROMPT_VERSION = "classify-v1"


@dataclass
class ChatClassifier:
    llm: LocalChatModel
    labels: type[enum.Enum]
    batch_size: int = 16
    op: str = "classify"
    traits: Traits = Traits(locality="in_process", egress=False, deterministic=True, requires=frozenset({"cuda"}))
    profile: Profile = field(default_factory=lambda: Profile(source="declared: in-process, no metered spend; latency unmeasured", usd_per_call=0.0))

    @property
    def name(self) -> str:
        return f"chat:{self.llm.model_id.split('/')[-1]}"

    @property
    def version(self) -> str:
        return PROMPT_VERSION

    def accepts(self, request: Request) -> bool:
        return request.op == "classify" and request.target is self.labels and isinstance(request.subject, str)

    def run(self, requests: Sequence[Request]) -> list[Output]:
        allowed = {m.value: m for m in self.labels}
        system = (
            "You label customer-support messages for a card and banking app. "
            "Reply with exactly one label from the list, copied verbatim, and nothing else. "
            "If none of the labels fits, reply: unknown\n\nLabels:\n" + "\n".join(allowed)
        )
        outs: list[Output] = []
        for i in range(0, len(requests), self.batch_size):
            chunk = requests[i : i + self.batch_size]
            for reply in self.llm.complete([(system, f"Message: {r.subject}\nLabel:") for r in chunk]):
                label = reply.strip().strip("`'\".").split()[0] if reply.strip() else ""
                if label in allowed:
                    outs.append(Output(allowed[label]))
                elif label.lower() == "unknown":
                    outs.append(Output(Unknown("model_declined")))
                else:
                    outs.append(Output(Unknown("unparseable_output", reply[:80])))
        return outs
