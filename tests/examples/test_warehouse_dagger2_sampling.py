"""The proposed source weighting is explicit, deterministic and comprehensive."""

import pytest

from examples.robotics.warehouse.learned_policy.dagger2_sampling import (
    balanced_episode_batches,
)


def test_equal_sources_cover_original_episodes_without_inventing_data():
    batches = balanced_episode_batches(70, 12, seed=42, epoch=0)
    assert len(batches) == 18
    assert batches == balanced_episode_batches(70, 12, seed=42, epoch=0)
    assert batches != balanced_episode_batches(70, 12, seed=42, epoch=1)
    assert {i for b in batches for source, i in b if source == "original"} == set(
        range(70)
    )
    for batch in batches:
        assert sum(source == "original" for source, _ in batch) == 4
        assert sum(source == "corrective" for source, _ in batch) == 4
        assert all(0 <= i < (70 if source == "original" else 12) for source, i in batch)


def test_empty_source_fails():
    with pytest.raises(ValueError):
        balanced_episode_batches(70, 0, seed=42, epoch=0)
