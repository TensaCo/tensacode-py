"""Hierarchical image features, learned layer by layer from unlabeled images.

The owner's direction: vision should be *hierarchical, learnable feature recognition*,
not icon templates. Each layer here learns a vocabulary from what the layer below
produces, without labels and without backpropagation:

* **layer 1** — small image patches, contrast-normalised and whitened, clustered by
  k-means: the centroids come out as oriented edges and colour blobs (Coates & Ng,
  *An analysis of single-layer networks in unsupervised feature learning*, 2011);
* **layer n+1** — neighbourhoods of the pooled layer-n map, clustered again: each
  centroid is a *composition* of lower features at relative positions (parts of
  edges, then parts of parts), in the spirit of learned compositional hierarchies
  (Fidler & Leonardis 2007) and of stacking k-means layers (Coates & Ng 2011b).

Encoding is the "triangle" activation: how much closer a patch is to a centroid than
its average distance to all centroids, floored at zero — sparse, and cheap.

This is numpy only: no network, no gradient, no GPU. Dot products appear in the
distance computations and nowhere else. ``numpy`` comes with the ``learned`` extra.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def patches(maps: np.ndarray, size: int, stride: int = 1) -> tuple[np.ndarray, tuple[int, int]]:
    """All ``size``x``size`` windows of ``maps`` (N, H, W, C) -> (N*h*w, size*size*C), and (h, w)."""
    n, h, w, c = maps.shape
    oh, ow = (h - size) // stride + 1, (w - size) // stride + 1
    s = maps.strides
    view = np.lib.stride_tricks.as_strided(
        maps, shape=(n, oh, ow, size, size, c), strides=(s[0], s[1] * stride, s[2] * stride, s[1], s[2], s[3]), writeable=False)
    return view.reshape(n * oh * ow, size * size * c), (oh, ow)


def sample_patches(maps: np.ndarray, size: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` random ``size``x``size`` windows, gathered without building every window."""
    count, h, w, c = maps.shape
    i = rng.integers(0, count, n)
    y = rng.integers(0, h - size + 1, n)
    x = rng.integers(0, w - size + 1, n)
    dy, dx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    rows = maps[i[:, None, None], y[:, None, None] + dy, x[:, None, None] + dx]  # (n, size, size, c)
    return rows.reshape(n, size * size * c)


def pool(maps: np.ndarray, k: int) -> np.ndarray:
    """Sum-pool (N, H, W, C) over non-overlapping k x k blocks (edges that do not fit are dropped)."""
    n, h, w, c = maps.shape
    h2, w2 = h // k, w // k
    return maps[:, : h2 * k, : w2 * k].reshape(n, h2, k, w2, k, c).sum(axis=(2, 4))


@dataclass
class Layer:
    """One learned vocabulary: patch size, how many features, and how its input is pooled first."""

    size: int
    k: int
    pool_before: int = 1
    whiten: bool = True
    eps: float = 10.0  # contrast-normalisation regulariser: 10 at the pixel scale (0-255), small above
    centroids: np.ndarray | None = field(default=None, repr=False)
    mean: np.ndarray | None = field(default=None, repr=False)
    zca: np.ndarray | None = field(default=None, repr=False)

    def _normalise(self, x: np.ndarray) -> np.ndarray:
        x = x - x.mean(axis=1, keepdims=True)
        return x / np.sqrt(x.var(axis=1, keepdims=True) + self.eps)

    def fit(self, x: np.ndarray, rng: np.random.Generator, *, iterations: int = 15, random: bool = False) -> None:
        """k-means on ``x`` (samples, dims). ``random=True`` keeps random centroids: the control."""
        x = self._normalise(x.astype(np.float32))
        if self.whiten:
            self.mean = x.mean(axis=0)
            cov = np.cov(x - self.mean, rowvar=False).astype(np.float32)
            d, v = np.linalg.eigh(cov)
            self.zca = (v @ np.diag(1.0 / np.sqrt(np.maximum(d, 0) + 0.1)) @ v.T).astype(np.float32)
            x = (x - self.mean) @ self.zca
        c = x[rng.choice(len(x), self.k, replace=False)].copy()
        if not random:
            for _ in range(iterations):
                assign = np.argmin(_sqdist(x, c), axis=1)
                for j in range(self.k):
                    members = x[assign == j]
                    if len(members):
                        c[j] = members.mean(axis=0)
                    else:
                        c[j] = x[rng.integers(len(x))]
        self.centroids = c.astype(np.float32)

    def encode(self, x: np.ndarray) -> np.ndarray:
        x = self._normalise(x.astype(np.float32))
        if self.whiten:
            x = (x - self.mean) @ self.zca
        d = np.sqrt(np.maximum(_sqdist(x, self.centroids), 0))
        return np.maximum(0, d.mean(axis=1, keepdims=True) - d).astype(np.float32)


def _sqdist(x: np.ndarray, c: np.ndarray) -> np.ndarray:
    return (x * x).sum(1, keepdims=True) - 2 * x @ c.T + (c * c).sum(1)[None, :]


@dataclass
class Hierarchy:
    """Layers learned bottom-up; ``describe`` pools the top layer into a fixed-length vector."""

    layers: list[Layer]
    grid: int = 2  # the top map is pooled into grid x grid regions

    def maps(self, images: np.ndarray, upto: int | None = None, batch: int = 250) -> np.ndarray:
        out = []
        for i in range(0, len(images), batch):
            m = images[i: i + batch].astype(np.float32)
            for layer in self.layers[: upto if upto is not None else len(self.layers)]:
                if layer.pool_before > 1:
                    m = pool(m, layer.pool_before)
                x, (h, w) = patches(m, layer.size)
                m = layer.encode(x).reshape(len(m), h, w, layer.k)
            out.append(m)
        return np.concatenate(out)

    def fit(self, images: np.ndarray, rng: np.random.Generator, *, samples: int = 100_000, random_layers: set[int] = frozenset()) -> None:
        """Learn each layer from samples of the maps the layers below produce."""
        for i, layer in enumerate(self.layers):
            below = self.maps(images, upto=i) if i else images.astype(np.float32)
            if layer.pool_before > 1:
                below = pool(below, layer.pool_before)
            layer.fit(sample_patches(below, layer.size, samples, rng), rng, random=i in random_layers)

    def describe(self, images: np.ndarray, batch: int = 250) -> np.ndarray:
        """Top-layer map pooled into ``grid`` x ``grid`` regions, computed batch by batch
        so the full maps (gigabytes for a large vocabulary) are never held at once."""
        out = []
        for i in range(0, len(images), batch):
            top = self.maps(images[i: i + batch], batch=batch)
            n, h, w, k = top.shape
            g = self.grid
            hs, ws = max(1, h // g), max(1, w // g)
            out.append(pool(top[:, : hs * g, : ws * g], hs).reshape(n, -1) if hs == ws else top.reshape(n, -1))
        return np.concatenate(out)
