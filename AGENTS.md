# Repository workflow

Work on `main` only. Do not create or switch to branches or isolated worktrees.
Preserve existing work and coordinate ownership when working in parallel.
The owner authorizes pushing completed, verified milestones to `origin/main`.
Report verification and limitations accurately.

# Architecture

Public API contracts and developer guidance live in `docs/README.md`.
Keep documentation focused on current library usage; design notes and
implementation history are preserved in Git.

Operations live under `tensorcode.ops.{vec,text,graph}` and follow a common callable
convention. Developers name their cognitive roles. Tools compose public operations.
Concrete vector encoders and decoders belong in `ops.vec.encode` and
`ops.vec.decode`; backend machinery belongs in `_internal.vec`. Keep public class
and persistence identities independent of backend filenames. Encoder vector sides
use `output_space`, decoder vector sides use `input_space`; foundation imports
remain lazy. Do not add backend-named public modules.
Tools own their model components and support configuration construction plus Hugging Face/local pretrained loading. Runtime composition, memory and action infrastructure live under tensorcode.runtime. Graph operations remain explicit symbolic stubs; do not restore callback or neural graph implementations without owner direction.

Tracing and training are independent of agent harnesses. Do not create domain-specific
schemas or implicit semantic policies in the general core.

Distinguish supplied models, authored policies and graph fixtures from capabilities
learned or inferred from real inputs. Report actual behavior and remaining gaps.
Do not claim trace capture makes arbitrary Python or remote models differentiable.
Do not restore the removed image-to-claim API, first-reader execution default,
bundled semantic seeds, or legacy compatibility paths.

The old implementation and historical reports are preserved privately at
https://github.com/JacobFV/old-tensorcode-2026-09-20 (checkpoint 716056b).

# Public operation construction

Public operations take JSON configuration, never supplied executable models in
constructors. Learned operations own weights and persist complete data-only
artifacts. Explicit `from_module` (vec) and `from_model` (text) are advanced
integration factories; arbitrary supplied implementations cannot be silently
reconstructed or discarded during persistence. Reject obsolete configuration
fields. Keep backend adapters private; graph operations remain symbolic stubs.
