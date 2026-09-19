"""Results: append-only, one row per (task, subject, split), with what is needed to believe it."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

from .core import commit

STORE = Path(__file__).parents[1] / "results" / "assay.jsonl"


def record(task_id: str, subject_id: str, split: str, metrics: Mapping[str, Any], *,
           dataset: str, license: str, data_digest: str = "", items: Iterable[Mapping] = (),
           self_authored: bool = False, notes: str = "") -> dict:
    row = {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "task": task_id, "subject": subject_id, "split": split,
           "metrics": dict(metrics), "dataset": dataset, "license": license, "data_digest": data_digest,
           "commit": commit(), "self_authored": self_authored, "notes": notes, "items": list(items)}
    STORE.parent.mkdir(parents=True, exist_ok=True)
    with STORE.open("a") as f:
        f.write(json.dumps(row) + "\n")
    return row


def rows() -> list[dict]:
    if not STORE.exists():
        return []
    return [json.loads(line) for line in STORE.read_text().splitlines() if line.strip()]


def latest(task_id: str, subject_id: str | None = None, split: str | None = None) -> dict | None:
    found = [r for r in rows() if r["task"] == task_id
             and (subject_id is None or r["subject"] == subject_id)
             and (split is None or r["split"] == split)]
    return found[-1] if found else None
