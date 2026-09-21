# TensorCode documentation

TensorCode separates callable operations, owned trainable models, application
runtime state and durable training experience. Start with the
[quickstart](quickstart.md) for a complete offline model lifecycle.

| Guide | Contents |
|---|---|
| [Pretrained checkpoints](pretrained.md) | Hosted tool catalog, pinned loading, scope and publication |
| [Quickstart](quickstart.md) | Install, initialize, collect feedback, train, save and reload |
| [Operations](operations.md) | Vector/message contracts, encoders, decoders and symbolic graph interfaces |
| [Tools](tools.md) | Owned models, shared workspace, pretrained artifacts and sessions |
| [Training](training.md) | Tool objectives, tracing, replay and resumable checkpoints |
| [Examples](../examples/README.md) | Learning agents and applications with explicit inputs |
| [Validation](validation.md) | Measured behavior, pretrained provenance and remaining gaps |
| [Evaluation records](results/README.md) | Machine-readable measurements |
| [Troubleshooting](troubleshooting.md) | Loading, gradients, replay and session errors |

## Public boundaries

`tensorcode.ops.{vec,llm,graph}` contains callable operations. Developers assign
names for cognitive roles; the same transform can participate in interpretation,
revision or response formulation. Representation contracts matter: equal tensor
shapes do not make independently trained encoders interchangeable.

`tensorcode.tools` contains owned PyTorch models: Chatbot, Investigator, Planner,
Decision and Scene. Scene combines image patches and text through the shared
workspace to rank supplied descriptions; it does not construct symbolic scene
graphs. Constructors initialize all
parameters; `from_pretrained` loads complete model artifacts from a local directory
or the Hugging Face Hub. An external foundation model can bootstrap training, but
its inherited competence does not establish that a new workspace has learned.

`tensorcode.runtime` contains explicit state and control-flow utilities. Policies,
candidate hypotheses, candidate plans and outcome feedback remain supplied inputs.
The general core does not supply a domain ontology or implicit action authority.

`tensorcode.tracing` and `tensorcode.training` capture operation dependencies and
train supported local tensor paths independently of an agent harness. Tracing does
not make arbitrary Python, remote model calls or discrete choices differentiable.
Graph operations currently declare symbolic interfaces without implementations.
