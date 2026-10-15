# Algorithms across substrates

`tensorcode.algorithm` represents a bounded tensor algorithm independently of how
it executes. Author the algorithm once, compile it to a symbolic executor, a
native differentiable circuit, or a scheduled attention machine, train its explicit
parameters, and export the updated algorithm. The JSON artifact is shared with
TypeScript's `tensorcode/algorithm`.

Install the current checkout with `pip install -e '.[algorithm]'`. The symbolic
executor requires NumPy. Importing this entry point does not import PyTorch;
compiling neural substrates does. The existing scalar `tensorcode.program` API
and its artifact format remain separate and unchanged.

## Compile an algorithm

```python
from tensorcode.algorithm import compile, compare, convert

algorithm = compile('''def policy(position, velocity, target, gain):
    error = target - position
    return {"torque": (gain * error - 2.5 * velocity).clip(-8, 8)}
''', inputs={"position": (20,), "velocity": (20,), "target": (20,)},
     parameters={"gain": 30.0})

symbolic = algorithm.compile("alg0")
circuit = algorithm.compile("alg1")
attention = algorithm.compile("alg2")
case = {"position": [0.0] * 20, "velocity": [0.0] * 20,
        "target": [0.1] * 20}

assert compare(symbolic, attention, [case])["passed"]
updated_symbolic = convert(attention, "alg0")
print(updated_symbolic(case)["torque"])
```

The frontend parses the source AST. It never executes supplied Python. Supported
source includes assignments, arithmetic, comparisons, conditional expressions,
literal-range loops, and the documented tensor methods. Loops unroll at compile
time. Arbitrary calls, imports, recursion, while loops, exceptions and host effects
are rejected. Conditional expressions and `where` are **eager**: both branches
must be valid. This differs from the scalar compiler's lazy selection.

Use `AlgorithmBuilder` for explicit graphs, graph composition and attention.
Each variable has a statically known shape. Inputs require exactly those shapes;
there are no implicit batch dimensions. All computation uses the artifact's
`float32` or `float64` dtype. Comparisons produce numeric zero/one values and have
no learned surrogate derivative.

## Execution contracts

| Target | Implementation | Contract |
|---|---|---|
| `alg0` | NumPy reference interpreter | Declared tensor operations and explicit state |
| `alg1` | Native PyTorch computation graph | Same operations, native autograd, backend numerical tolerance |
| `alg2`, hard routing | Scaled dot-product attention over operand registers, followed by declared local operations | Exact operand selection; backend numerical tolerance |
| `alg2`, soft routing | Finite structural penalties plus contextual QK logits | Approximate execution; validate before use |
| `alg2_bias` | Relations projected into existing attention through learned or supplied bindings | Structural prior, **not** an execution guarantee |

ALG.2 hard execution is a scheduled neural/symbolic attention machine. Its
instruction schedule and local arithmetic are supplied by the compiler. It is not
an assertion that arbitrary algorithms have been converted into an ordinary
fixed-depth pretrained transformer. All compiled machines expose `manifest()` to
make these distinctions inspectable.

Each instruction's argument roles become separate directed relations. Query row
`i`, key column `j` means instruction `i` reads source `j`. In the attention
machine, previous register values are padded into a value memory, and actual
scaled dot-product attention retrieves each argument. Hard masks allow exactly one
source per argument. Soft routing allows other earlier registers with a finite
penalty; it can change behavior even if the graph is correct.

`context=` may supply a `[node_count, context_dim]` tensor to an attention machine.
With hard single-source masks its content logits cannot change operand selection.
With soft routing they can. The default context is zero. This explicit binding
must not be confused with a model learning what every algorithm variable means.

## Structural compilation into a pretrained transformer

The prior target implements:

```text
bias[head] = sum_role strength[head, role] * Pquery @ Arole @ Pkey.T
```

`Arole` comes from the authored algorithm. The query and key binding networks are
independent and include an edge-free null slot. They are initially untrained.
Users may train them with their own task objective or supply explicit probability
distributions over algorithm nodes plus null. Relation strengths start at zero.

```python
from tensorcode.algorithm import StructuralBias, BiasedMultiheadAttention

bias_model = algorithm.compile("alg2_bias", hidden_dim=768, heads=12)
# query_latents/key_latents: [batch, tokens, hidden_dim]
# bias = bias_model(query_latents, key_latents)  # [batch, heads, queries, keys]
# Add bias to scaled QK logits before softmax, retaining the original legal mask.

# For an existing torch.nn.MultiheadAttention:
# adapter = BiasedMultiheadAttention.from_module(pretrained_attention, algorithm)
# output, weights = adapter(query_latents, key_latents, values)
```

`from_module` copies existing projections, accepts batch-first equal-width
attention without extra key/value tokens or dropout, and freezes those projections
by default. `from_projections(q, k, v, out, algorithm, heads=...)` accepts four
standard pretrained Linear projections using `1/sqrt(head_dim)` scoring.
This factory does not automatically patch a model or invent positional semantics.
Pass any original attention and padding masks explicitly. Extra new tokens are
not inserted, so the original softmax denominator is preserved at zero bias.
Numerical backend changes can still introduce floating-point differences.

`checkpoint()` and `restore()` persist the complete adapter, including its owned
projection weights, binding weights, relation strengths and dtype, as data-only
objects. They do not serialize Python code or pickle executable modules.

Run the pretrained integration check using a locally cached pinned checkpoint:

```bash
python examples/pretrained_attention_bias.py \
  --image docs/results/algorithm-robotics/robot.png \
  --output /tmp/pretrained-bias-check.json
```

The [pretrained attention example](../examples/pretrained_attention_bias.py)
checks a pinned local CLIP checkpoint on an actual image. Its measured result is
function preservation at zero bias, gradient flow, and changed output at nonzero
bias. It does **not** show that CLIP understands or executes a robotics policy.

## Train and convert back

Both neural execution targets are `torch.nn.Module` instances. Parameters declared
in the algorithm are registered parameters; ordinary optimizers update them.
`artifact()` exports their current values, rather than stale initialization values.

```python
import torch

optimizer = torch.optim.Adam(attention.parameters(), lr=0.01)
optimizer.zero_grad()
result = attention(case)["torque"]
loss = (result - torch.ones(20, dtype=torch.float64)).square().mean()
loss.backward()
optimizer.step()

trained = attention.artifact()
restored = type(trained).from_json(trained.to_json())
standalone_source = trained.to_python("policy")
```

`to_python` emits readable standalone NumPy code returning `(outputs, next_state)`.
Artifact loading never evaluates this source. `freeze()` replaces explicit
parameters with constants. `optimize()` removes unreachable instructions while
preserving the public input/state/parameter signature.

A hard attention machine can round-trip through ALG.0 and ALG.1. A soft machine
cannot silently export its nominal algorithm as an equivalent implementation:
`artifact()` and exact conversion reject that request. Its `checkpoint()` preserves
its approximation settings. Any symbolic replacement needs separate validation.

## State and recurrence

```python
from tensorcode.algorithm import AlgorithmBuilder

b = AlgorithmBuilder(name="filter")
x = b.input("sample", (3,))
state = b.state("estimate", [0., 0., 0.])
alpha = b.parameter("alpha", 0.2)
next_estimate = state + alpha * (x - state)
b.update(state, next_estimate)
filter_algorithm = b.build(next_estimate)

module = filter_algorithm.compile("alg2")
outputs, next_state = module.step({"sample": [1., 2., 3.]})
outputs, next_state = module.step({"sample": [2., 3., 4.]}, state=next_state)
```

State updates are simultaneous. Neither executor mutates caller state; failures
produce no partial state commit. `unroll(sequence, state=...)` on Python modules
retains the state computation graph for backpropagation through time. Optional
`detach_every` requests explicit truncated gradients. There is no hidden simulator
or event loop in the framework.

## Import supported trained computation

`from_module` imports current weights from supported PyTorch Sequential, Linear,
ReLU, Sigmoid, Tanh, last-axis LayerNorm, Softmax and Identity modules.
`from_attention` imports supported multi-head self-attention, including QK, OV,
softmax and an optional causal mask. Unsupported modules fail closed.

These factories do not require TensorCode compiler provenance. They reproduce the
supported neural computation as an algorithm artifact. They do **not** discover a
compact semantic algorithm from an arbitrary pretrained model. Layer normalization
is represented explicitly; it is not silently linearized.

## Validation and boundaries

`compare(reference, candidate, cases, rollout=True)` compares both outputs and
state across the supplied trajectory, allowing each implementation to carry its own
state forward. It reports tolerance, number of cases, maximum error and the first
counterexample. Empty evaluations are rejected. `validate_replacement` rejects a
candidate when comparison fails. These reports are empirical, not continuous-domain
proofs, and do not establish task quality by themselves.

Artifacts reject unsupported versions, unknown node fields, invalid dependencies,
shape inconsistencies, nonfinite data and oversized individual tensors. Execution
rejects fully masked attention rows and nonfinite intermediate results. Limits are
10,000 nodes, rank six, and one million elements per tensor; dense attention
register memory has a separate budget. These bounds are not a sandbox for executing
untrusted workloads. No arbitrary host callbacks are available.

## Robotics policy example and measured result

Install `pip install -e '.[robotics]'`, then run:

```bash
MUJOCO_GL=egl OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python examples/robotics_policy.py --output /tmp/my-robot-policy
```

The output directory must be new. The example authors a 20-joint fixed-base robot
in MuJoCo and an impedance policy with model-based gravity compensation. It trains
60 actuator-correction coefficients through ALG.2 from a supplied synthetic
calibration dataset, exports the policy, and runs ALG.0/1/2 physics rollouts.
It writes portable JSON, standalone Python, an attention checkpoint, trajectory
records, a report and an optional GIF replay.

In the recorded six-second run, baseline joint RMS error was **0.010393 rad**;
with the learned correction it was **0.000576 rad**, matching across all three
substrates. On a separate target phase with 25% stronger actuator disturbance,
error was **0.011928 rad** baseline versus **0.002518 rad** corrected. Held-out
calibration MSE was `3.57e-9`, compared with `0.294` for a zero residual.

This demonstrates simulated joint tracking and learning/conversion mechanics.
It does not demonstrate locomotion, learned visual grounding, real-system
identification, hardware safety, or hard real-time execution. Measured ALG.2
latency exceeded the 5 ms simulation timestep on some iterations; the simulator
advances offline. See the [recorded report](results/algorithm-robotics/report.json),
[portable policy](results/algorithm-robotics/policy.json), and
[replay](results/algorithm-robotics/robot.gif).
