# Consistent owned operation conventions

User approved full implementation with no backward compatibility. Work on main;
push verified milestones. Graph remains unimplemented. Heavy inference/training
only on GB10. Existing authored policies remain explicit, no semantic seeds.

## Contracts

All public operations use op(value, *, context=None). Primary constructor takes
JSON configuration, not a model/module/callback. Parameter-free operations accept
optional config where meaningful. Unknown/obsolete fields fail instead of being
ignored. Public classes remain at ops.<representation>.<operation>.

Learned operations own their registered parameters and reconstruct from data-only
artifacts through save_pretrained/from_pretrained. from_foundation initializes
supported native pretrained architectures explicitly. Random heads/bridges remain
labeled untrained. Native latent models support context={'latents':[...]} with
strict Space/mask/batch validation. No generic claim of learned cognition.

Explicit advanced factories from_module (vec) / from_model (text) accept supplied
implementations. They are not old constructor aliases. Arbitrary executable
modules/providers cannot promise reconstructible artifacts; reject unsupported
artifact saves rather than silently discard behavior. Known owned configurations
must round trip all parameters, settings, tokenizer/processor assets, and losses.

Pure operations (text serialization, vector top-k and graph stubs) need no weights.
They use config-based constructors and safe config persistence where implemented.
Graph calls and unavailable pretrained semantics still raise NotImplementedError.

Existing tool-owned nn adapters and non-owning language realization move/remain
private. Their state registration stays coherent. All current callers/tests/docs
migrate; no public legacy constructors, generic fallback aliases or hidden model
selection. Existing historic reports remain immutable. Version 0.4.0a3.

## Ownership/tasks

- [x] Vector worker: public transform.py/classify.py/score.py/decode.py plus new
  _internal/vec owned backend modules. Config-owned linear/MLP and native transformer
  Transform; trainable Classify/Score heads; coherent generic Decode or explicit
  modality-only surface (coordinate parent). Supported from_foundation and complete
  local/Hub artifacts, target-only objectives, stable public objective identities.
  Explicit from_module for advanced injection, truthful unsupported-save errors.
  Own tests/vec for these ops except encoder/latent/configuration tests parent.
- [x] Text worker: all ops/text and new _internal/text. Config-owned native local
  seq2seq Transform/Classify/Score/Decide/Retrieve with full model artifact lifecycle
  and differentiable teacher-forced objectives; retain strict output validation,
  async/batch/tracing. Explicit from_model(provider) advanced integration preserves
  remote behavior. Dependency-free public imports via lazy owned backend. Pure
  encode/decode use JSON config. Move old tool-only Decode private and update
  Chatbot import. Own tests/text and new owned-text lifecycle tests.
- [x] Consumers worker: migrate src/tools and _internal users of old vector
  Transform to private TensorAdapter, tests outside tests/vec,text to explicit
  factories/new configs. Do not modify chatbot decoder import (text worker).
- [x] Docs/examples worker: current docs/examples use new constructors, explicit
  advanced factories where required; new default owned vector lifecycle example.
  Preserve historical result JSON. Describe architecture support and artifact
  limits accurately, no compatibility guides preserving removed constructors.
- [x] Parent: move original raw Transform into private adapter before worker edits;
  own VocabularyEncoder/PatchEncoder config+owned save/load, pure vector selectors
  and graph stub conventions, shared parameter-free config artifact machinery.
  Update root exports if needed, AGENTS/version. Encoder/configuration/latent tests.
- [x] Independent reviews, all tests, build/isolated core imports, artifact/loss
  restart assertions, docs syntax/links, commit and push, verify CI.

## Review focus

No model args in public constructors. No arbitrary artifact-directed imports.
Config/model weights survive owned reload. Targets never enter source context.
Masks/context and untrained projections are explicit. Remote calls stay external
and non-differentiable. Pure operations don't manufacture learned behavior.

## Verification

Final CPU suite: 596 passed, 1 skipped (second-device CUDA assertion).
Wheel and sdist build passed; isolated wheel imports preserve dependency-free core.
Owned vector and text operation tests cover safe artifacts, durable experiences,
and exact optimizer checkpoint continuation. Owned lifecycle example completed
training, weight reload parity, and resumed updates. Review corrected native
text mode races, missing instruction identity, stale image metadata persistence,
and duplicate foundation allocation. No high-throughput training or new pretrained
quality claim is part of this milestone. Graph remains explicitly unimplemented.
