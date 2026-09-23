# Changelog

All notable changes to the `tensorcode` Python package. Versions follow
[PEP 440](https://peps.python.org/pep-0440/); alpha releases may break APIs.
See [updating development code](docs/migration.md) for import changes.

## 0.4.0a4 (unreleased)

### Changed

- `Chatbot`, `Investigator`, `Decision`, `Planner` and `Scene` reject unknown
  or obsolete configuration fields with a `ValueError` that names each unknown
  field and lists the valid ones, for example
  `Unknown Planner configuration fields: ['colour']; valid fields: [...]`.
  Nested `Chatbot` `cognition` fields and each `Scene` mode (ranking or
  `language`) are checked the same way. Because `from_pretrained` reconstructs
  through the constructor, a saved artifact with an unknown field is rejected.
  Owned retrieval and response-quality models now report the same message.

### Fixed

- The operation configuration of a foundation-backed ranking encoder (for
  example Electra or BERT in `Planner.from_foundation`) is now JSON: it
  describes the native configuration and tensor schemas instead of walking
  Transformers' non-JSON module attributes.
- The private message-sequence memory store no longer imports a module that
  does not exist.

## 0.4.0a3

### Added

- Text operations (`Classify`, `Decide`, `Score`, `Retrieve`) accept
  `decoding='likelihood'`, which scores every authored alternative in one
  encoder pass and returns a full distribution instead of generated JSON.
- `tensorcode.ops.text.ask` / `aask` ask several named structured questions
  about the same messages; a shared external model that implements
  `complete_questions` answers them in one request.
- `tensorcode.integrations.JevModel`, an explicit HTTP adapter for typed
  classification, decision and rubric-score requests.
- `tensorcode.training` calibration utilities: `TemperatureCalibration`,
  `evaluate_calibration` and `fit_threshold`, fitted only on held-out scores.
- `readout='output_encoding'` for owned text and image encoders: an owned
  trainable token appended after the context.
- Investigator `verification_scope='joint'` checks a hypothesis against the
  combined evidence and keeps each per-source check.
- `Investigator.new_cognitive_session(...)` / `load_cognitive_session(...)`,
  `Planner.new_executor(...)` and `tools.actions.action_loop(...)` factories.

### Changed

- Public boundaries are smaller. The session, memory, cognition, execution and
  tracing machinery moved under `tensorcode._internal`. Use root
  `tensorcode.trace`, `Trace`, `InputRef` and `OutputRef`;
  `training.Trainer.from_tool(...)` / `Trainer.from_ops(...)`; and
  `training.load_experience(...)`. `tensorcode.runtime` and
  `tensorcode.tracing` are removed without compatibility aliases.
- Public operations are constructed from JSON configuration. They own their
  weights and persist with `save_pretrained` / `from_pretrained`. Supplied
  implementations use the explicit `from_module` (vector) and `from_model`
  (text) factories.
- The Chatbot workspace uses a bounded (`relative_rms_bounded`) memory update.
  Artifacts must match the current architecture exactly.
- Public records, operations, tools and trainer properties now have
  docstrings. Every `ops.vec` submodule declares `__all__`, so internal
  helpers no longer leak through star imports.
- Package metadata now includes classifiers, keywords and project, docs and
  changelog URLs. The sdist includes only the documented top-level paths.
- Tests of the repository's internal `.development` experiments skip, rather
  than fail, when run from the unpacked sdist, which does not ship that tree.

### Removed

- `DecisionPipeline`, `runtime.*` session/memory/storage classes, the free
  checkpoint functions and `training.ToolTrainer`. See the
  [migration table](docs/migration.md).

## 0.4.0a2

- Vector APIs are organized by operation. Concrete encoders live in
  `tensorcode.ops.vec.encode` and decoders in `tensorcode.ops.vec.decode`.
  Backend modules are private.
- Alpha model artifacts were recreated with the current public operation
  identities.

## 0.4.0a1

- Added owned pretrained vector operations: transformer `TextEncoder` and
  `ImageEncoder`, a seq2seq `TextDecoder` and a latent-diffusion
  `ImageDecoder` (`tensorcode[diffusion]`).

## 0.3.0

- Added the owned pretrained tool lifecycle (`save_pretrained`,
  `from_pretrained`, Hugging Face Hub loading) and the learned evidence
  workspace shared by Chatbot, Investigator, Planner, Decision and Scene.
  Built distributions are attached to the
  [GitHub release](https://github.com/TensaCo/tensacode-py/releases/tag/v0.3.0).
