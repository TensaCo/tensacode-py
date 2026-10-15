# Compiled numerical programs

`tensorcode.program` compiles bounded Python numerical functions into the same
portable scalar SSA circuit used by the TypeScript SDK. Programs combine exact
computation, owned parameters, explicit gradient policies and transactional state.
This API is separate from the symbolic `tensorcode.ops.graph` namespace.

## How it works

Write a supported numerical function, or construct one with `Builder`. Compilation
turns it into an **intermediate representation (IR)**: a list of numerical
instructions and their dependencies. Each intermediate value is assigned once
(**SSA**, static single assignment). Branches select values, and bounded loops
expand into instructions. No examples, fitted weights or pretrained model are
needed to reproduce the authored rule.

You can run that circuit, check it against an independent function, and then add
parameters to learn. Training changes only those declared parameters; it does
not automatically rewrite the source or discover a new algorithm. Operation
tracing records calls between existing operations and is a separate mechanism.

The examples describe this source checkout; use its development installation
when trying the compiler APIs.

## Compile and check a rule

```python
from tensorcode.program import compile, int32

def choose(a):
    return 10 if a > 4 else 12

program = compile(choose, inputs={"a": int32(min=-128, max=127)})
assert program.run({"a": 5}) == {"result": 10}
assert program.verify()["cases"] == 256
artifact = program.to_json()
```

A callable must have inspectable source. Source strings containing one function
are also accepted; supply an independent reference to `verify` for these.
Decorators, closures, defaults, arbitrary calls and host side effects are rejected.
The Python frontend accepts arithmetic, comparisons, `if`/`else`, conditional
expressions, assignments/augmented assignments, early returns, literal arrays,
indexing, `relu`/`sigmoid`, and literal `range` loops. Defaults bound expansion to
1,024 iterations and 100,000 nodes. Unsupported/unbounded syntax is rejected.
The frontend rejects potentially inexact binary64 integer arithmetic; narrow
input ranges instead of silently losing Python's arbitrary-precision semantics.
Arithmetic on floating inputs follows the declared finite numeric contract.

## Python API

```python
from tensorcode.program import Builder, Runtime, Program, float64

b = Builder(semantics="hybrid", surrogate="sigmoid")
x = b.input("x", float64())
w = b.parameter("weight", 0.0)
p = b.build(b.select(x > 4, w * x, 0))
p.run({"x": 5})
p.gradients({"x": 5})
model = p.to_tensor_module()  # imports optional torch only here
```

The module accepts scalar floating PyTorch tensors and returns tensors by output
name. Its owned parameters work with native optimizers and state dictionaries.
The bridge extracts scalar values and executes the CPU reference evaluator, then
propagates its first-order gradients through PyTorch. It is not a vectorized
PyTorch or CUDA implementation of the circuit.
`model.artifact()` includes updated weights. For existing trace/Trainer workflows,
construct `ProgramOperation(artifact)` from `tensorcode.program.operation`, then
use `Trainer.from_ops`. The operation rejects state and host calls for pure replay.

The builder supports overloaded `+`, `-`, `*`, `/`, unary minus and ordered
comparisons. Use `b.op("eq", x, y)` for equality, `b.select(c, yes, no)` for
selection, and `b.gather(values, index)` for a fixed list. Symbolic values cannot
be used as Python truth values. Additional APIs are `one_hot`, `gather_one_hot` (checked exactly-one selection),
`affine`, `sparse_affine`, `scan`, `call`,
`constant`, `state`, `update`, and `include(program, inputs, prefix="region")`.
`freeze(names=None)` converts parameters to constants. Arrays are explicitly
scalarized; arbitrary tensor shapes are not compiled.

`program.step(inputs, state=state)` returns `(outputs, next_state)`. `Runtime`
commits state only after successful execution, bounds event steps, and supports
`snapshot()`, `Runtime.restore(snapshot)` and `replay(events)`. Snapshots share the
TypeScript format. `Program.from_json(text)` loads shared IR. The standard-library
compiler/runtime does not import torch; training integration requires the `vec`
extra or a compatible torch installation.

Custom calls are bound with `bindings={"name": (forward, vjp)}`. `vjp(args, result)`
returns one scalar derivative per argument. A forward-only binding may use `None`
for its VJP, but cannot silently participate in differentiation. Bindings are not
serialized. They must be pure numerical functions during differentiation.

`verify(reference=None, *, cases=None, max_cases=100000, atol=0, rtol=0)` passes named
inputs as keyword arguments to the independent reference. It exhaustively checks
finite integer domains or marks supplied cases non-exhaustive. Fixed array returns
are mapped to string-indexed output names. `verified_replacement` accepts a
candidate only after exhaustive finite-domain agreement, with frozen parameters,
no state/calls and matching signatures. This is a validation gate, not a learned
program synthesizer or a proof over real numbers.

`optimize()` performs checked constant folding and dead-code elimination;
`schedule()` exposes dependency layers; `analyze(program)` reports intervals and
potential invalid operations. Multiply/add chains are packed into affine kernels with preserved rounding
points. `sparse_affine` stores only supplied coefficients in ascending index
order, without a dense matrix. No floating reassociation or SIMD vectorization
is performed. Runtime domain checks remain authoritative.

## Add a learned correction

This example needs `tensorcode[vec]`. The fixed rule returns 10 at x=5. The
combined program starts at 10, and one optimizer step moves its output toward
the target 11 by updating only `correction`. The comparison and constants remain
fixed. The resulting model intentionally differs from the original rule.

```python
import torch
from tensorcode.program import Builder

rules = Builder()
a = rules.input("a")
algorithm = rules.build(rules.select(a > 4, 10, 12))
b = Builder()
x = b.input("x")
correction = b.parameter("correction", 0.0)
region = b.include(algorithm, {"a": x}, prefix="algorithm")
program = b.build(region["result"] + correction * x)
model = program.to_tensor_module()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
optimizer.zero_grad()
output = model({"x": torch.tensor(5.0, dtype=torch.float64)})["result"]
(output - 11).square().backward()
optimizer.step()
trained_artifact = model.artifact()
```

## State and events

```python
from tensorcode.program import Builder, Runtime, int32

b = Builder()
event = b.input("event", int32(min=0, max=1))
count = b.state("count", 0, int32(min=0, max=100))
b.update(count, count + event)
runtime = Runtime(b.build(count), max_steps=100)
assert runtime.dispatch({"event": 1}) == {"result": 0}
saved = runtime.snapshot()
resumed = Runtime.restore(saved)
assert resumed.dispatch({"event": 1}) == {"result": 1}
```

Each output sees the old state. Updates commit simultaneously after every check
passes; a failed event changes neither state nor the step count. The snapshot
contains the circuit, state and consumed step budget, so resuming does not reset
the execution limit. Callbacks and host resources are not serialized.

## Shared semantics and portability

Domains are `int32(min=..., max=...)`, `boolean()`, `float32(min=..., max=...)`, and
`float64(min=..., max=...)`. Inputs are finite numbers; boolean wires use 0/1. Integer
inputs are checked, never silently truncated or wrapped. Arithmetic defaults to
binary64, like ordinary Python floats, with no reassociation or
fused multiply-add. `dtype='float32'` explicitly requests rounding at each circuit
node; this differs from ordinary Python float arithmetic. Python arbitrary-size
integer arithmetic is not the circuit representation. Input
domains describe input representation independently of computation dtype.
NaN, infinity, division by zero, invalid indices and domain overflow are errors.
Signed zero is preserved by artifacts serialized as objects in memory, but ordinary
JSON number serialization does not preserve negative zero; do not depend on its
sign across JSON transport. Source division by zero is outside the finite contract.

Exact comparison is a comparison instruction, not a ReLU approximation. Thus
`4.5 > 4` produces 1 with a floating input domain. Exact selection executes only
the chosen branch, including in hybrid forward execution.

| Semantics | Forward | Backward |
| --- | --- | --- |
| `exact` | Checked numerical program | Ordinary piecewise derivatives; comparisons/index decisions have zero gradient |
| `relaxed` | Sigmoid comparisons, Gaussian equality, continuous selection | Derivative of the relaxation |
| `hybrid` | Same hard comparisons and selection as exact | Explicit `surrogate="sigmoid"` policy |

`temperature` controls relaxations. A vector–Jacobian product (VJP) carries an output gradient back to its inputs.
Hybrid selection's VJP uses both branch
values. If an inactive branch is undefined, exact forward can succeed while the
requested hybrid derivative fails. ReLU's derivative at zero is defined as zero.
These are first-order policies, not claims of mathematical differentiability at
hard decision boundaries or useful learning signals for every program.

## Practical limits

`graph()` returns inspectable IR data, not a graphical renderer. Verification
compares executions against a supplied reference; finite cases do not prove a
continuous domain or every possible stateful event sequence. Interval analysis
is diagnostic rather than a proof of equivalence.

Event state belongs to `Runtime`. Tensor modules do not carry event state from
one forward call to the next, and this integration does not provide training
through an event history. ProgramOperation deliberately accepts only stateless,
call-free programs.

| Situation | What to do |
| --- | --- |
| A function cannot be inspected or a closure is rejected | Use self-contained source text or Builder; pass inputs explicitly |
| Verification has no reference | Supply an independent function when loading an artifact or compiling text |
| A fractional value is rejected by int32 | Use a floating domain if fractions are intended |
| Loop expansion exceeds the budget | Reduce the bound or explicitly configure the expansion and node limits |
| Hybrid gradients fail although forward succeeds | Check both branch values; the surrogate derivative may need the inactive branch |
| A tensor has more than one element | Select scalar values explicitly; this API is not a general tensor compiler |

## Artifact version 1

The root has `format: "tensorcode.program"`, `version: 1`, `semantics`, `dtype`,
`surrogate`, `temperature`, `nodes`, `outputs` and `updates`. Each node has a
contiguous topological `id`, opcode `op`, preceding IDs in `args`, scalar
`shape: []`, checked `domain`, `differentiability`, and source location (or null).
Bindings add `name`, literals/initial values add `value`; included nodes can
specify their own `semantics`, `surrogate` and `temperature`. Outputs map names to node IDs;
updates map state-node IDs to next-value IDs. There is no executable source in
an artifact. Loading validates IDs, arity, domains, semantics and state targets.

Python executes the portable CPU reference and integrates with PyTorch.
The TypeScript SDK also emits WebAssembly and stateless float32 WGSL. WebGPU
results are checked against the CPU using explicitly supplied tolerances; GPU
floating-point arithmetic is not promised bit-exact. These backends do not imply
that arbitrary Python functions, foundation models or host tools become neural
circuits. Source locations describe authored code, not learned capabilities.

Validation: `.venv/bin/python -m pytest tests/test_program.py tests/training -q`.
The TypeScript program suite also sends artifacts to Python and runs
Python-compiled artifacts through TypeScript and WebAssembly.
