"""Load declared episode splits, never randomly split neighboring frames."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from .contract import SCHEMA, actions, features


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_episode(path):
    """Rows are paired pre-action observations and actually issued commands."""
    path = Path(path)
    document = json.loads(
        gzip.decompress(path.read_bytes())
        if path.suffix == ".gz"
        else path.read_bytes()
    )
    if document.get("schema") != SCHEMA:
        raise ValueError("Unrecognized or legacy episode: action provenance required")
    rows = document["steps"]
    if not rows:
        raise ValueError("Empty episode")
    times = np.asarray([row["t"] for row in rows], dtype=float)
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Episode times must be finite and strictly increasing")
    x = np.stack([features(row["observation"]) for row in rows])
    y = np.stack([actions(row["action"]) for row in rows])
    return x, y, document


def load_manifest(path):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    splits = {name: [] for name in ("train", "validation", "test")}
    seen_paths, seen_hashes, seen_groups = set(), set(), {}
    for split, episodes in splits.items():
        for entry in manifest[split]:
            episode = (path.parent / entry["path"]).resolve()
            digest = sha256(episode)
            if digest != entry["sha256"]:
                raise ValueError(f"Episode hash mismatch: {episode}")
            if episode in seen_paths or digest in seen_hashes:
                raise ValueError("Duplicate episode across declared splits")
            # Concurrent robots or rerenders of the same physical run are one group.
            group = entry["physical_run_id"]
            if group in seen_groups and seen_groups[group] != split:
                raise ValueError("Physical run leaks across episode splits")
            seen_paths.add(episode)
            seen_hashes.add(digest)
            seen_groups[group] = split
            x, y, document = load_episode(episode)
            episodes.append((x, y, document, entry))
    if not splits["train"] or not splits["validation"]:
        raise ValueError("Nonempty train and validation episode splits required")
    return manifest, splits
