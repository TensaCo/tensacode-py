# Operation boundary refactor implementation plan

Approved design: public API follows tensorcode.ops.<representation>.<operation>.
The owner authorizes greenfield implementation, commits/pushes on main, and no
additional approval stages. This file records execution rather than a public guide.

## Contract

- Public concrete vector classes are defined in encode.py and decode.py; their
  class identities and artifact manifests use those paths. Public aliases at
  vec root refer to the same classes. Private backends live in _internal/vec.
- Encoder vector side is output_space; decoder vector side is input_space.
  Pretrained encoders use readout='sequence' or 'pooled'. Native text pooled
  readout is masked mean; ViT pooled readout is CLS. No new readout training.
- Encoder and decoder context is an ordered {'latents': [...]} prefix, validated
  against declared context_space for encoders and input_space for decoders.
  Encoder contexts use explicit native embedding-width spaces; no implicit
  semantic alignment is claimed. Preserve source masks and native gradients.
- Both decoder constructors use bridge='linear' or 'identity'; modality-specific
  generation controls and losses remain explicit. Core imports stay optional.
- VocabularyEncoder/PatchEncoder are deliberately supported specialized encode
  operations with output_space. Tool-only SequenceEncoder moves private.
- Old backend-named module imports and alpha operation identities are removed.
  Existing tool state layouts remain unchanged. No legacy module aliases.
  Older alpha operation artifacts/traces require recreation from source; document
  this break rather than silently reinterpreting binding fingerprints.
- Graph stays stubbed. No heavy compute on host; no new training required.

## Tasks and ownership

- [x] Text/vision worker: move text_model and vision_model implementations into
  _internal/vec/text.py and vision.py; import optional transformers on construction;
  standardize encoder output_space/readout and implement real masked latent-prefix
  context for text. Own tests/vec/test_pretrained_{text,vision}.py. Tests first for
  new signatures/context, then preserve all existing native/artifact regressions.
- [x] Diffusion worker: move diffusion.py into _internal/vec/diffusion.py; normalize
  bridge argument/config; private objective identifies public owner, not backend.
  Own tests/vec/test_image_decode.py; verify masks, loss, native round trips.
- [x] Parent: move mechanical backend/SequenceEncoder; define public concrete
  classes; central exports; update mechanical contracts/callers/tests, stable
  objective binding identities, version 0.4.0a2. Boundary tests must reject old
  modules, verify public class identities and optional imports.
- [x] Docs/examples worker: update current developer guides/examples to imports,
  output_space/readout/bridge/latent context; retain historical reports unchanged.
- [x] Independent review, focused and full CPU tests, wheel/core imports, build,
  checkpoint manifest inspection. Commit/push coherent milestone and verify CI.

## Review focus

1. Importing vec operations must not load transformers/diffusers.
2. Artifacts and replay objectives must identify public operations.
3. Context mask and batch mismatch must not silently change behavior.
4. Native weights/tokenizers/processors remain exact through artifacts.
5. Existing owned tools keep parameter registration/weakref ownership intact.

## Verification and decisions

- 501 CPU tests passed; one GPU-only test skipped. Tiny native ALBERT exposes the
  distinction between embedding width and hidden output width; context validation
  and native embedding lookup now follow the actual encoder module.
- All four pretrained operation manifests/model cards and restored concrete types
  have public operation identities. Private objectives use an internal explicit
  persistence identity hook (owner path + objective role); dataclass identity and
  default operation identity are unchanged. Changed configurations still reject
  replay even when a private objective implementation moves.
- Standalone backend-named alpha artifacts are deliberately incompatible; no old
  module aliases or unsafe fingerprint rewrites. Existing owned tool state shapes
  and weakref ownership remain unchanged and tool round trips pass.
- Backend module imports do not load Transformers or Diffusers. Core imports keep
  PyTorch optional. Public VocabularyEncoder/PatchEncoder remain explicitly named
  mechanical encoders under the encode operation, not pretrained fallbacks.
- Full build, installed core wheel check, example syntax, guide links and diff
  checks complete before commit. CI status is recorded by the pushed GitHub run.
- No new heavy training/inference; historical GB10 reports retain their source
  commits. Public guide identifies this limit.
