"""Frozen equal-source episode batches for a proposed corrective fine-tune.

This module launches no fit. Recurrent state must be reset once per episode batch,
then carried chronologically through all chunks (detached only for gradients).
"""

import math
import random


def balanced_episode_batches(old_count, corrective_count, *, seed, epoch):
    """Four original + four corrective episodes, cycling shuffled pools.

    An epoch covers every original episode at least once. Corrective episodes
    repeat by design; repetition is exposure weighting, not more demonstrations.
    """
    if old_count < 1 or corrective_count < 1:
        raise ValueError("Both episode sources must be nonempty")
    rng = random.Random(seed + epoch)
    pools = [[], []]
    counts = [old_count, corrective_count]

    def draw(source):
        if not pools[source]:
            pools[source] = list(range(counts[source]))
            rng.shuffle(pools[source])
        return pools[source].pop()

    return [
        [("original", draw(0)) for _ in range(4)]
        + [("corrective", draw(1)) for _ in range(4)]
        for _ in range(math.ceil(old_count / 4))
    ]
