# Program compilation milestone — 2026-10-15

Owner-requested work: make deterministic programs, trainable parameters and explicit
gradient policies composable in the Python and TypeScript SDKs. This is compiler
and execution infrastructure; it does not establish improved cognitive tool
quality or shared latent-space learning.

Implemented in `tensorcode.program` / `tensorcode/program`:

- Shared version-1, data-only scalar SSA artifacts with domains, source locations,
  state updates, owned parameters and explicit exact/relaxed/hybrid policies.
- Bounded Python/TypeScript source frontends, circuit builders, exact comparisons,
  lazy selection, fixed-array gather, checked one-hot selection and ordered scans.
- Reference execution, custom VJPs, frozen-region composition, registered native
  tensor modules, replayable operations and existing Trainer/checkpoint support.
- Constant folding, dead-code elimination, affine-chain fusion retaining rounding
  points and host boundaries, sparse affine construction and dependency schedules.
- Transactional state snapshots/replay, finite-domain equivalence verification and
  a verified replacement gate for externally proposed programs.
- TypeScript WebAssembly emission and stateless float32 WGSL execution with explicit
  CPU tolerance verification for every GPU result.

Acceptance evidence:

1. M1: exhaustive branching over all 256 inputs in [-128,127], plus real-valued
   comparison behavior at 4.5. CPU and emitted WASM agree.
2. M2: static loops, ordered prefix sums, simultaneous finite-state updates,
   snapshot restoration, and failure atomicity.
3. M3: gradients through compiled regions reach owned parameters; supplied scalar
   regression targets train through the existing trace/Trainer path. Native
   checkpoints restore operation weights and trainer step counts.
4. M4: TypeScript artifacts execute in Python; Python-compiled artifacts execute in
   TypeScript/WASM. A real headless Chromium run passed CPU, WASM, state, WebGPU
   comparison/selection and WebGPU affine execution. The GPU adapter was software
   SwiftShader, not a hardware throughput benchmark.

Validation run: TypeScript typecheck/build; program/training/nn suite (341 tests
before the final policy/boundary regression additions); Python program/training
suite (71 tests before those additions). The final focused program suites cover
those additional regressions. Generated docs and 1,246 internal links passed.

Limits are public in `docs/program.md`: scalarized circuits, a deliberately bounded
source subset, no arbitrary host-source compilation, no general tensor-shape
frontend, no higher-order gradients, no unbounded-loop proof, no automatic learned
synthesizer, and no claim that GPU floating point is portable bit-exact. GPU output
verification adds reference execution cost. Structural proposals remain external;
finite-domain replacement verification is implemented. Ordinary JSON transport
has the documented negative-zero limitation.

The authored circuits, exact reference functions, finite input domains and scalar
training targets in these tests are supplied fixtures. No pretrained model,
held-out cognitive evaluation or learned reasoning capability is claimed.
