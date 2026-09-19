"""The frozen held-out prompt sets (eval/heldout) as assay items."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

from ..core import Dataset, Item, Prompt

DIR = Path.home() / ".cache" / "tensorcode" / "heldout"
LICENSES = {"desktop_gui": "Apache-2.0", "shell_files": "GPL-3.0", "screen_questions": "CC-BY-4.0",
            "image_questions": "CC-BY-4.0", "conversation_facts": "MIT", "general_knowledge": "CC-BY-SA-3.0",
            "arithmetic": "MIT", "open_ended": "CC-BY-SA-3.0", "ambiguous": "CC-BY-SA-3.0",
            "multi_step": "CC-BY-4.0", "owner": "owner"}
SOURCES = {"desktop_gui": "OSWorld", "shell_files": "NL2Bash", "screen_questions": "ScreenQA",
           "image_questions": "VQA v2", "conversation_facts": "LongMemEval", "general_knowledge": "NQ-open + WebQuestions",
           "arithmetic": "GSM8K", "open_ended": "Dolly-15k", "ambiguous": "AmbigNQ + ClariQ",
           "multi_step": "Mind2Web + OSWorld", "owner": "the owner's own prompt"}


def available() -> bool:
    return (DIR / "dev.jsonl").exists()


def digest(split: str) -> str:
    path = DIR / f"{split}.jsonl"
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16] if path.exists() else ""


def _gold(row: dict):
    ref = row.get("reference") or {}
    category = row["category"]
    if category == "arithmetic":
        return [str(ref.get("final", "")).replace(",", "")]
    if category == "image_questions":
        answers = [a for a in (ref.get("answers") or [ref.get("answer")]) if a]
        return answers
    if category in ("general_knowledge", "screen_questions"):
        return [a for a in (ref.get("answers") or []) if a]
    if category == "conversation_facts":
        return [str(ref.get("answer"))] if ref.get("answer") else []
    return ref or None


def load(category: str):
    def loader(split: str) -> Sequence[Item]:
        path = DIR / f"{split}.jsonl"
        if not path.exists():
            return []
        out = []
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["category"] != category:
                continue
            images = ()
            if row.get("image") and (DIR / row["image"]).exists():
                images = ((DIR / row["image"]).read_bytes(),)
            history = tuple(str(m["content"])[:2000] for turn in (row.get("history") or [])
                            for m in (turn.get("messages") or []) if m.get("role", "user") == "user" and m.get("content"))
            out.append(Item(row["id"], Prompt(row["text"], images, history), _gold(row),
                            {"category": category, "reference": row.get("reference")}))
        return out

    return loader


def dataset(category: str) -> Dataset:
    return Dataset(name=SOURCES.get(category, category), license=LICENSES.get(category, "see MANIFEST"),
                   url="eval/heldout/MANIFEST.json", load=load(category), available=available,
                   fetch_hint="uv run eval/heldout/build.py")
