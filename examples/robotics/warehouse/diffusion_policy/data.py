"""Episode-local causal histories and masked action chunks, with train-only statistics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from ..learned_policy.contract import input_statistics
from ..learned_policy.data import load_episode, sha256


def load_training_manifest(path):
    """Validate split identity without opening or scoring final/test episode bytes."""
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    splits = {"train": [], "validation": []}
    paths, hashes, groups = set(), set(), {}
    for split in ("train", "validation", "test"):
        for entry in manifest[split]:
            source = (path.parent / entry["path"]).resolve()
            digest, group = entry["sha256"], entry["physical_run_id"]
            if source in paths or digest in hashes:
                raise ValueError("Duplicate episode across declared splits")
            if group in groups and groups[group] != split:
                raise ValueError("Physical run leaks across episode splits")
            paths.add(source)
            hashes.add(digest)
            groups[group] = split
            if split == "test":
                continue
            if sha256(source) != digest:
                raise ValueError(f"Episode hash mismatch: {source}")
            x, y, _ = load_episode(source)
            splits[split].append((x, y))
    if not splits["train"] or not splits["validation"]:
        raise ValueError("Nonempty train and validation splits required")
    return manifest, splits


def normalization(episodes):
    x = np.concatenate([row[0] for row in episodes])
    y = np.concatenate([row[1] for row in episodes])
    mean, scale = input_statistics(x)
    action_mean, action_scale = y.mean(0), np.maximum(y.std(0), 0.01)
    action_mean[-2:], action_scale[-2:] = 0, 1
    return mean, scale, action_mean, action_scale


class Windows(Dataset):
    def __init__(self, episodes, statistics, history=8, horizon=16, stride=4):
        if min(history, horizon, stride) < 1:
            raise ValueError("History, horizon and stride must be positive")
        mean, scale, action_mean, action_scale = statistics
        self.episodes = [
            ((x - mean) / scale, (y - action_mean) / action_scale) for x, y in episodes
        ]
        self.history, self.horizon = history, horizon
        self.indices = [
            (episode, start)
            for episode, (x, _) in enumerate(episodes)
            for start in range(0, len(x), stride)
        ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        episode, start = self.indices[index]
        x, y = self.episodes[episode]
        past = np.clip(np.arange(start - self.history + 1, start + 1), 0, len(x) - 1)
        future = np.arange(start, start + self.horizon)
        return (
            x[past],
            y[np.minimum(future, len(y) - 1)],
            (future < len(y)).astype(np.float32),
        )
