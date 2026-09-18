"""Learned implementations: a trained request parser and a trained extractive answerer.

Requires the ``learned-neural`` extra (torch, transformers). Both are registered the same
way every other implementation is, so the runtime routes to them by declared traits and
the trace records which one answered.

The parser turns an utterance into ``ParsedRequest(act, slots)``; the answerer turns a
question plus passages into ``Answer(text, evidence)``. Each loads an artifact directory
that carries its own label space and its calibrated abstention threshold, so the code
here cannot drift from what was trained. Confidence is the calibrated probability that
the output is correct, measured on a held-out split named in the artifact; below the
threshold the implementation abstains with ``Unknown`` rather than guessing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

from ..outcomes import Score, Unknown
from ..runtime import Output, Profile, Request, Traits


# --------------------------------------------------------------- what they answer


@dataclass(frozen=True)
class ParsedRequest:
    """What an utterance asks for: an act and its slots, as the assistant's procedures expect."""

    act: str
    slots: Mapping[str, Any] = field(default_factory=dict)
    confidence: float | None = None
    speech_act: str | None = None  # command / question / self_disclosure / world_statement / other

    def __repr__(self) -> str:
        return f"{self.act}({', '.join(f'{k}={v!r}' for k, v in sorted(self.slots.items()))})"


@dataclass(frozen=True)
class QuestionOverPassages:
    """A question and the passages that may contain its answer."""

    question: str
    passages: tuple[tuple[str, str], ...]  # (title, sentence)


@dataclass(frozen=True)
class Answer:
    text: str
    evidence: tuple[tuple[str, str], ...] = ()  # the passages the span came from
    confidence: float | None = None


# ------------------------------------------------------------------- the models


class RequestParserModel(nn.Module):
    """One encoder, four kinds of head: the act, a BIO tagger for span slots, closed choices, flags."""

    def __init__(self, encoder_id: str, n_acts: int, n_tags: int, closed_sizes: Sequence[int], n_flags: int) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_id)
        h = self.encoder.config.hidden_size
        self.act = nn.Linear(h, n_acts)
        self.tags = nn.Linear(h, n_tags)
        self.closed = nn.ModuleList([nn.Linear(h, n) for n in closed_sizes])
        self.flags = nn.Linear(h, n_flags)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        pooled = hidden[:, 0]
        return {
            "act": self.act(pooled),
            "tags": self.tags(hidden),
            "closed": [head(pooled) for head in self.closed],
            "flags": self.flags(pooled),
            "pooled": pooled,
        }


class SpanAnswererModel(nn.Module):
    """Start/end over the passage, plus one head for "the passage does not answer this"."""

    def __init__(self, encoder_id: str) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_id)
        h = self.encoder.config.hidden_size
        self.span = nn.Linear(h, 2)
        self.answerable = nn.Linear(h, 2)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        kw = {"token_type_ids": token_type_ids} if token_type_ids is not None else {}
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kw).last_hidden_state
        start, end = self.span(hidden).split(1, dim=-1)
        return {"start": start.squeeze(-1), "end": end.squeeze(-1), "answerable": self.answerable(hidden[:, 0])}


# ------------------------------------------------------------------ inference


def _device(prefer: str | None = None) -> str:
    if prefer:
        return prefer
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class NeuralRequestParser:
    """A trained utterance -> (act, slots) parser, as a ``parse`` implementation."""

    artifact: Path
    device: str | None = None
    batch_size: int = 64
    threshold: float | None = None  # overrides the artifact's calibrated threshold
    name: str = "neural-request-parser"
    op: str = "parse"

    def __post_init__(self) -> None:
        self.artifact = Path(self.artifact)
        self.config = json.loads((self.artifact / "config.json").read_text())
        self.device = _device(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.artifact / "tokenizer"))
        closed = self.config["closed_heads"]
        self.model = RequestParserModel(
            self.config["encoder"], len(self.config["acts"]), len(self.config["tags"]),
            [len(v) for _, v in closed], len(self.config["flags"]),
        )
        state = torch.load(self.artifact / "weights.pt", map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()
        self.version = str(self.config.get("version", "1"))
        self._threshold = self.threshold if self.threshold is not None else float(self.config.get("threshold", 0.0))
        self.traits = Traits(locality="in_process", egress=False, deterministic=True,
                             requires=frozenset({"cuda"} if self.device.startswith("cuda") else set()))
        q = self.config.get("quality", {})
        self.profile = Profile(source=self.config.get("measured_on", "unmeasured"), quality=q,
                               latency_ms_p50=self.config.get("latency_ms_p50"), latency_ms_p95=self.config.get("latency_ms_p95"),
                               usd_per_call=0.0, peak_memory_mb=self.config.get("peak_memory_mb"))

    # -- runtime contract

    def accepts(self, request: Request) -> bool:
        return request.op == "parse" and request.target is ParsedRequest and isinstance(request.subject, str)

    def run(self, requests: Sequence[Request]) -> list[Output]:
        texts = [r.subject for r in requests]
        parsed = self.parse(texts)
        outs: list[Output] = []
        for got in parsed:
            score = Score(got.confidence or 0.0, "probability", self.profile.source or "unmeasured")
            if got.act == "unknown":
                reason = "not_a_request" if got.speech_act == "world_statement" else "not_a_known_request"
                detail = ("this states something rather than asking for anything"
                          if got.speech_act == "world_statement" else f"no act fits this utterance (p={got.confidence:.2f})")
                outs.append(Output(Unknown(reason, detail), score))
            elif (got.confidence or 0.0) < self._threshold:
                outs.append(Output(Unknown("below_threshold", f"p={got.confidence:.3f} < {self._threshold:.3f}", (got,)), score))
            else:
                outs.append(Output(got, score))
        return outs

    # -- the work

    @torch.inference_mode()
    def parse(self, texts: Sequence[str]) -> list[ParsedRequest]:
        out: list[ParsedRequest] = []
        for i in range(0, len(texts), self.batch_size):
            out.extend(self._parse_batch(list(texts[i : i + self.batch_size])))
        return out

    @torch.inference_mode()
    def embed(self, texts: Sequence[str]) -> torch.Tensor:
        """Sentence embeddings from the same encoder, for associative recall over claims."""
        vecs = []
        for i in range(0, len(texts), self.batch_size):
            batch = self.tokenizer(list(texts[i : i + self.batch_size]), return_tensors="pt", padding=True,
                                   truncation=True, max_length=self.config["max_length"]).to(self.device)
            vecs.append(self.model(batch["input_ids"], batch["attention_mask"])["pooled"].float().cpu())
        return torch.cat(vecs) if vecs else torch.zeros(0, self.model.encoder.config.hidden_size)

    def _parse_batch(self, texts: list[str]) -> list[ParsedRequest]:
        cfg = self.config
        enc = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                             max_length=cfg["max_length"], return_offsets_mapping=True)
        offsets = enc.pop("offset_mapping")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        got = self.model(enc["input_ids"], enc["attention_mask"])
        act_p = got["act"].softmax(-1).cpu()
        tag_id = got["tags"].argmax(-1).cpu()
        closed_id = [c.argmax(-1).cpu() for c in got["closed"]]
        flag_on = (got["flags"] > 0).cpu()

        whole_input = frozenset(cfg.get("whole_input_slots", ()))
        results = []
        for row, text in enumerate(texts):
            act = cfg["acts"][int(act_p[row].argmax())]
            confidence = float(act_p[row].max())
            spans = _decode_spans(cfg["tags"], tag_id[row].tolist(), offsets[row].tolist(), text,
                                  enc["attention_mask"][row].cpu().tolist())
            slots: dict[str, Any] = {}
            allowed = frozenset(cfg["act_slots"].get(act, []))
            for slot, value in spans.items():
                if slot in allowed and slot != "place":
                    slots[slot] = value
            closed = {name: vocab[int(closed_id[k][row])] for k, (name, vocab) in enumerate(cfg["closed_heads"])}
            if "place" in allowed:
                kind = closed.get("place_kind", "none")
                if kind == "span" and "place" in spans:
                    slots["place"] = spans["place"]
                elif kind not in ("none", "span"):
                    slots["place"] = kind
            if "target" in allowed and closed.get("target_kind", "none") == "@it":
                slots["target"] = "@it"  # "read it": the thing acted on is whatever was last touched
            if "name" in allowed and closed.get("name_canon", "none") != "none" and "name" not in slots:
                slots["name"] = closed["name_canon"]  # "make a readme": a name supplied by convention
            if act == "info" and closed.get("info_topic", "none") != "none":
                slots["topic"] = closed["info_topic"]
            if "unit" in allowed and closed.get("unit", "none") != "none":
                slots["unit"] = closed["unit"]
            for slot in allowed & whole_input:
                slots[slot] = text  # nothing to tag: the procedure reads the whole request back out
            if "aspect" in allowed:
                slots["aspect"] = closed.get("aspect", "none")
            for k, flag in enumerate(cfg["flags"]):
                if flag in allowed:
                    slots[flag] = bool(flag_on[row][k])
            speech_act = closed.get("speech_act")
            if speech_act == "world_statement":
                # a remark about the world is not a request; acting on it would be a wrong action,
                # so the speech act overrules whatever the act head preferred
                act, slots = "unknown", {}
            results.append(ParsedRequest(act, slots, confidence, speech_act))
        return results


def _decode_spans(tags: Sequence[str], ids: Sequence[int], offsets: Sequence[Sequence[int]], text: str,
                  mask: Sequence[int]) -> dict[str, str]:
    """First contiguous B/I run per slot, mapped back to the original characters."""
    spans: dict[str, tuple[int, int]] = {}
    current: tuple[str, int, int] | None = None
    for position, tag_id in enumerate(ids):
        if position >= len(offsets) or not mask[position]:
            continue
        start, end = offsets[position]
        if start == end:  # special token
            continue
        tag = tags[tag_id]
        if tag == "O":
            if current:
                spans.setdefault(current[0], (current[1], current[2]))
                current = None
            continue
        prefix, slot = tag.split("-", 1)
        if current and current[0] == slot and prefix == "I":
            current = (slot, current[1], end)
        else:
            if current:
                spans.setdefault(current[0], (current[1], current[2]))
            current = (slot, start, end)
    if current:
        spans.setdefault(current[0], (current[1], current[2]))
    return {slot: text[a:b].strip() for slot, (a, b) in spans.items() if text[a:b].strip()}


@dataclass
class NeuralAnswerer:
    """A trained extractive answerer over passages, as a ``parse`` implementation."""

    artifact: Path
    device: str | None = None
    batch_size: int = 16
    threshold: float | None = None
    max_length: int = 384
    name: str = "neural-span-answerer"
    op: str = "parse"

    def __post_init__(self) -> None:
        self.artifact = Path(self.artifact)
        self.config = json.loads((self.artifact / "config.json").read_text())
        self.device = _device(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.artifact / "tokenizer"))
        self.model = SpanAnswererModel(self.config["encoder"])
        self.model.load_state_dict(torch.load(self.artifact / "weights.pt", map_location="cpu", weights_only=True))
        self.model.to(self.device).eval()
        self.version = str(self.config.get("version", "1"))
        self._threshold = self.threshold if self.threshold is not None else float(self.config.get("threshold", 0.0))
        self.traits = Traits(locality="in_process", egress=False, deterministic=True,
                             requires=frozenset({"cuda"} if self.device.startswith("cuda") else set()))
        self.profile = Profile(source=self.config.get("measured_on", "unmeasured"), quality=self.config.get("quality", {}),
                               latency_ms_p50=self.config.get("latency_ms_p50"), latency_ms_p95=self.config.get("latency_ms_p95"),
                               usd_per_call=0.0, peak_memory_mb=self.config.get("peak_memory_mb"))

    def accepts(self, request: Request) -> bool:
        return request.op == "parse" and request.target is Answer and isinstance(request.subject, QuestionOverPassages)

    def run(self, requests: Sequence[Request]) -> list[Output]:
        answers = self.answer([r.subject for r in requests])
        outs = []
        for got in answers:
            score = Score(got.confidence or 0.0, "probability", self.profile.source or "unmeasured")
            if not got.text:
                outs.append(Output(Unknown("not_in_passage", f"no span answers this (p={got.confidence:.2f})"), score))
            elif (got.confidence or 0.0) < self._threshold:
                outs.append(Output(Unknown("below_threshold", f"p={got.confidence:.3f} < {self._threshold:.3f}", (got,)), score))
            else:
                outs.append(Output(got, score))
        return outs

    @torch.inference_mode()
    def answer(self, items: Sequence[QuestionOverPassages]) -> list[Answer]:
        out: list[Answer] = []
        for i in range(0, len(items), self.batch_size):
            out.extend(self._answer_batch(list(items[i : i + self.batch_size])))
        return out

    @torch.inference_mode()
    def _answer_batch(self, items: list[QuestionOverPassages]) -> list[Answer]:
        contexts = [" ".join(s for _, s in it.passages) for it in items]
        enc = self.tokenizer([it.question for it in items], contexts, return_tensors="pt", padding=True,
                             truncation="only_second", max_length=self.max_length, return_offsets_mapping=True)
        offsets = enc.pop("offset_mapping")
        seq_ids = [enc.sequence_ids(i) for i in range(len(items))]
        enc = {k: v.to(self.device) for k, v in enc.items()}
        got = self.model(**{k: v for k, v in enc.items() if k in ("input_ids", "attention_mask", "token_type_ids")})
        answerable = got["answerable"].softmax(-1)[:, 1].cpu()
        starts, ends = got["start"].cpu(), got["end"].cpu()

        results = []
        for row, item in enumerate(items):
            allowed = [i for i, s in enumerate(seq_ids[row]) if s == 1]
            if not allowed:
                results.append(Answer("", (), float(answerable[row])))
                continue
            s_log, e_log = starts[row][allowed], ends[row][allowed]
            s_p, e_p = s_log.softmax(-1), e_log.softmax(-1)
            best, span = 0.0, None
            top_s = torch.topk(s_p, k=min(20, len(allowed))).indices.tolist()
            top_e = torch.topk(e_p, k=min(20, len(allowed))).indices.tolist()
            for a in top_s:
                for b in top_e:
                    if b < a or b - a > 30:
                        continue
                    p = float(s_p[a] * e_p[b])
                    if p > best:
                        best, span = p, (allowed[a], allowed[b])
            if span is None:
                results.append(Answer("", (), float(answerable[row])))
                continue
            a, b = offsets[row][span[0]][0].item(), offsets[row][span[1]][1].item()
            text = contexts[row][a:b].strip()
            confidence = float(answerable[row]) * best ** 0.5
            evidence = tuple(p for p in item.passages if p[1] and text and p[1].find(text) >= 0)
            results.append(Answer(text if float(answerable[row]) >= 0.5 else "", evidence, confidence))
        return results
