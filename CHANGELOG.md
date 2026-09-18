# Changelog

No part of the API is stable before 1.0. Any 0.x release may change or remove anything;
this file lists what changed so upgrading is not guesswork.

## Unreleased

- `propose` is no longer exported from `tensorcode`. It is still available as
  `tensorcode.ops.propose` and is experimental: no example, test or evaluation uses it yet.

## 0.1.0a1 — 2026-09-18

First release of the revived package, published as `tensorcode`. The legacy `tensacode`
code (2023–2024, never published, never functional) is preserved at the git tag
`legacy-2024-11`; the PyPI name `tensacode` is a placeholder that installs `tensorcode`.
