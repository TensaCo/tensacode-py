"""Inference for the trained sentence selector, plus the truncation guard it needed.

The guard is here rather than in a test because it is a property of every evaluation that feeds
text to a fixed window: `eval/training/eval_span.py` handed ~40 sentences to a 384-token model
and reported the result as a capability measurement, when two thirds of the evidence had been
silently dropped. An evaluation that can truncate must count how often it did.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Truncation:
    """How much of what an evaluation handed to a model the model could actually see."""

    items: int = 0
    truncated: int = 0
    tokens_given: int = 0
    tokens_used: int = 0
    worst_dropped: int = 0

    def update(self, other: "Truncation") -> None:
        """Fold another count in, so a per-batch guard can report per-benchmark."""
        self.items += other.items
        self.truncated += other.truncated
        self.tokens_given += other.tokens_given
        self.tokens_used += other.tokens_used
        self.worst_dropped = max(self.worst_dropped, other.worst_dropped)

    def record(self, given: int, limit: int) -> None:
        self.items += 1
        self.tokens_given += given
        self.tokens_used += min(given, limit)
        if given > limit:
            self.truncated += 1
            self.worst_dropped = max(self.worst_dropped, given - limit)

    def report(self) -> dict:
        share = self.truncated / self.items if self.items else 0.0
        return {
            "items": self.items, "truncated": self.truncated, "share_truncated": round(share, 4),
            "mean_tokens_given": round(self.tokens_given / max(1, self.items), 1),
            "mean_tokens_seen": round(self.tokens_used / max(1, self.items), 1),
            "worst_case_tokens_dropped": self.worst_dropped,
            "verdict": ("clean" if not self.truncated else
                        f"{share:.0%} of items lost evidence to the window; accuracy here is not a capability measurement"),
        }


def truncation_of(tokenizer, pairs: list[tuple[str, str]], limit: int) -> Truncation:
    """Count, per item, whether (question, context) fits the model's window."""
    got = Truncation()
    for question, context in pairs:
        n = len(tokenizer(question, context, truncation=False)["input_ids"])
        got.record(n, limit)
    return got


@dataclass
class Selector:
    """Scores one (question, sentence) pair at a time, batched. Higher is more likely to be gold."""

    artifact: Path
    device: str | None = None
    batch_size: int = 256
    _loaded: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        import json

        import torch
        from torch import nn
        from transformers import AutoModel, AutoTokenizer

        self.artifact = Path(self.artifact)
        self.config = json.loads((self.artifact / "config.json").read_text())
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.artifact / "tokenizer"))
        encoder = AutoModel.from_pretrained(self.config["encoder"])
        head = nn.Linear(encoder.config.hidden_size, 1)

        class _Selector(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.encoder, self.head = encoder, head

            def forward(self, **enc):
                return self.head(self.encoder(**enc).last_hidden_state[:, 0]).squeeze(-1)

        self.model = _Selector()
        self.model.load_state_dict(torch.load(self.artifact / "weights.pt", map_location="cpu", weights_only=True))
        self.model.to(self.device).eval()
        self.max_length = int(self.config.get("max_length", 160))

    def score(self, question: str, sentences: list[tuple[str, str]]) -> list[float]:
        """One score per (title, sentence), in the order given."""
        import torch

        out: list[float] = []
        texts = [f"{t}. {s}" for t, s in sentences]
        with torch.inference_mode():
            for i in range(0, len(texts), self.batch_size):
                chunk = texts[i : i + self.batch_size]
                enc = self.tokenizer([question] * len(chunk), chunk, return_tensors="pt", padding=True,
                                     truncation=True, max_length=self.max_length).to(self.device)
                out.extend(torch.sigmoid(self.model(**enc)).cpu().tolist())
        return out

    def top(self, question: str, sentences: list[tuple[str, str]], k: int) -> list[tuple[str, str]]:
        """The k best sentences, returned in their original passage order."""
        scored = self.score(question, sentences)
        keep = {i for i, _ in sorted(enumerate(scored), key=lambda p: -p[1])[:k]}
        return [s for i, s in enumerate(sentences) if i in keep]
