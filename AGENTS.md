# Repository workflow

Work on `main` only. Do not create or switch to branches or isolated worktrees.
Preserve existing work and coordinate ownership when working in parallel.
The owner authorizes pushing completed, verified milestones to `origin/main`.
Report verification and limitations accurately.

# Architecture

The governing replacement architecture is
`docs/a-new-hope/1-tensorcode-architecture.md`, based on the owner's feedback in
`docs/a-new-hope/0-jacob-valdez-feedback.md`.

Operations live under `tensorcode.ops.{vec,llm,graph}` and follow a common callable
convention. Developers name their cognitive roles. Tools compose public operations.
Tracing and training are independent of agent harnesses. Do not create domain-specific
schemas or implicit semantic policies in the general core.

Distinguish supplied models, authored policies and graph fixtures from capabilities
learned or inferred from real inputs. Report actual behavior and remaining gaps.
Do not claim trace capture makes arbitrary Python or remote models differentiable.
Do not restore the removed image-to-claim API, first-reader execution default,
bundled semantic seeds, or legacy compatibility paths.

The old implementation and historical reports are preserved privately at
https://github.com/JacobFV/old-tensorcode-2026-09-20 (checkpoint 716056b).
