# TensorCode documentation

TensorCode separates callable operations, owned trainable models, application
runtime state and durable training experience. Start with the
[quickstart](quickstart.md) for a complete offline model lifecycle.

| Guide | Contents |
|---|---|
| [Pretrained checkpoints](pretrained.md) | Hosted tool catalog, pinned loading, scope and publication |
| [Quickstart](quickstart.md) | Install, initialize, collect feedback, train, save and reload |
| [Operations](operations.md) | Vector/message contracts, encoders, decoders and symbolic graph interfaces |
| [Pretrained vector models](latent-models.md) | Transformer encoding, text/diffusion decoding, training, canonical imports and owned configuration |
| [Tools](tools.md) | Owned models, shared workspace, pretrained artifacts and sessions |
| [Cognition](cognition.md) | Sourced evidence, generated proposals, verification, revision and action feedback |
| [Training](training.md) | Tool objectives, tracing, replay and resumable checkpoints |
| [Examples](../examples/README.md) | Learning agents and applications with explicit inputs |
| [Validation](validation.md) | Measured behavior, pretrained provenance and remaining gaps |
| [Evaluation records](results/README.md) | Machine-readable measurements |
| [Troubleshooting](troubleshooting.md) | Loading, gradients, replay and session errors |

## Public boundaries

`tensorcode.ops.{vec,text,graph}` contains callable operations. Developers assign
names for cognitive roles; the same transform can participate in interpretation,
revision or response formulation. Representation contracts matter: equal tensor
shapes do not make independently trained encoders interchangeable. Concrete vector
encoders live in `tensorcode.ops.vec.encode`, and decoders in
`tensorcode.ops.vec.decode`; root exports are conveniences. Backend modules are
private. See the [operation API](operations.md) and
[owned operation contracts](latent-models.md#public-paths-and-owned-configuration).

Learned operation constructors accept JSON configuration and own their model
parameters. Explicit `from_foundation` factories initialize supported pretrained
architectures; `save_pretrained`/`from_pretrained` persist owned artifacts.
Advanced vector `from_module` and text `from_model` factories integrate supplied
implementations with explicit reconstruction limits. Weightless operations accept
optional configuration; graph calls still raise `NotImplementedError`.

`tensorcode.tools` contains owned PyTorch models: Chatbot, Investigator, Planner,
Decision and Scene. Scene combines image patches and text through the shared
workspace to rank supplied descriptions or, with an owned language foundation,
produce explicitly unverified image interpretations. Neither mode constructs
symbolic scene graphs. Constructors initialize all
parameters; `from_pretrained` loads complete model artifacts from a local directory
or the Hugging Face Hub. An external foundation model can bootstrap training, but
its inherited competence does not establish that a new workspace has learned.

Tools construct their sessions and execution helpers. `tools.cognition` exposes
evidence and interpretation records; `tools.actions` exposes action callback
records and an explicit bounded-loop factory. Planner exposes structured plan
contracts and `new_executor(...)`. Evidence, policies and action implementations
remain explicit, while memory storage and state transitions stay internal.
Generated hypotheses are not source evidence; generated plans are not executable
action code. The general core supplies no domain ontology or implicit authority.

Root `tensorcode.trace()` and `tensorcode.training` capture operation dependencies and
train supported local tensor paths independently of an agent harness. Tracing does
not make arbitrary Python, remote model calls or discrete choices differentiable.
Graph operations currently declare symbolic interfaces without implementations.
