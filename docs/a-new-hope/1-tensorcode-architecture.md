# 1 — TensorCode, from scratch

*2026-09-20. Architecture specification consolidated from [Jacob's feedback](0-jacob-valdez-feedback.md) and the subsequent design conversation. The owner subsequently authorized the archive, destructive reset, and implementation. This describes the full intended architecture, not a claim that every capability ships in the first replacement milestone. Python examples specify proposed interfaces; the implementation report identifies the executable subset.*

## 1. Purpose

TensorCode is a Python library for composing cognitive operations into ordinary software and training the supported parts of that software from experience.

A developer should be able to add one learned judgment to a request handler, construct a multimodal cognitive architecture, or import a ready-to-use tool. All three should use the same underlying operations. None should require adopting a particular agent loop, domain ontology, global runtime, or workflow language.

The central abstraction is an operation: a callable object that performs a meaningful transformation or judgment. Operations can use learned models, explicit algorithms, or combinations of both. Their intermediate representations can be vectors, LLM messages, or symbolic graphs. Python supplies composition and control flow.

The library has four principal responsibilities:

1. **Operations:** small, composable capabilities with consistent calling conventions.
2. **Tools:** useful, importable compositions with sensible, explicit configurations.
3. **Tracing:** capture operation execution and data dependencies within a chosen scope.
4. **Training:** use feedback and those dependencies to improve components that support an applicable learning method.

Encoders, cognitive processing, memory, agents, and interfaces fit within this structure. There is no mandatory singleton cognitive core. A program's composition of operations is its cognitive architecture.

### Success looks like this

- A support router uses the same decision machinery that an agent uses to select an action.
- A programmer can name a generic transform `update_objective`, `merge_observations`, or `form_hypothesis` without requiring three new framework APIs.
- A tool works immediately with a documented model configuration, and its components can subsequently be replaced.
- A vector program retains native gradients across compatible operations.
- A trace can become a training example without treating its internal predictions as independent external inputs.
- Claims about learning are supported by measured changes on held-out inputs, not by the existence of a schema, trace, or model-shaped class.

## 2. Scope and architectural constraints

This is a fresh architecture, not a compatibility layer around the current agent implementation.

The replacement must not require grammar, speech-act, goal, scene, task-admission, or domain-specific schemas before a user can call a primitive. Domain structures belong in applications, tools, supplied knowledge, or optional specializations.

This does not prohibit structure. Identity, source evidence, alternatives, uncertainty, and revisability remain useful requirements. They should be provided where the application needs them rather than through a compulsory universal ontology.

The objective in [revival/36](https://github.com/JacobFV/old-tensorcode-2026-09-20/blob/716056ba2cbe66402f28dc764c66bdc2195f1c09/docs/revival/36-structured-cognitive-workspace.md)—building and revising interpretations from evidence before acting—remains a useful agent acceptance criterion. Its existing implementation sequence does not prescribe the new package structure. The prohibitions in [revival/39](https://github.com/JacobFV/old-tensorcode-2026-09-20/blob/716056ba2cbe66402f28dc764c66bdc2195f1c09/docs/revival/39-removing-implicit-semantic-authority.md) remain constraints: do not reintroduce removed semantic defaults through renamed helpers or compatibility options.

Explicit pretrained models are legitimate sources of capability. Their learned priors must be distinguished from library-authored policies and capabilities learned during a user's session. A convenience constructor must disclose what model, prompts, and policies it installs.

Non-goals for the foundational release:

- A universal autonomous agent or complete theory of cognition.
- Automatic differentiation through arbitrary Python, remote services, or real-world actions.
- A complete matrix of every operation in every representation before anything is usable.
- A distributed serving platform, scheduler, or mandatory graph editor.
- Domain knowledge silently bundled into general-purpose operations.

## 3. Package organization

```text
tensorcode/
    __init__.py
    ops/
        __init__.py       # public operation contracts and shared operand/result types
        base.py
        vec/
            __init__.py
            latent.py
            encode/
                text.py
                image.py
            decode/
                text.py
            transform.py
            classify.py
            decide.py
            score.py
            retrieve.py
        llm/
            __init__.py
            messages.py
            encode/
            decode/
            transform.py
            classify.py
            decide.py
            score.py
            retrieve.py
        graph/
            __init__.py
            representation.py
            encode/
            decode/
            transform.py
            classify.py
            decide.py
            score.py
            retrieve.py
    tools/
        decision/
            support_router.py
        agents/
            chatbot.py
    tracing/
        session.py
        graph.py
        storage.py
    training/
        supervision.py
        replay.py
        trainer.py
    integrations/        # provider/framework adapters, added as needed
    _internal/           # genuinely private implementation helpers
    utils/               # small conveniences, not a second architecture
```

This is a responsibility map. Create modules when they contain an implemented capability; do not scaffold empty implementations to make the tree symmetrical.

The first release should prefer one small module over a subpackage until an actual second implementation warrants the directory. In particular, do not create an abstract hierarchy of named cognitive acts simply to fill this tree.

Public imports follow `tensorcode.ops.<representation>.<operation>`. Representation packages may re-export common classes for shorter imports. Public extension contracts must be accessible from `tensorcode.ops`, even if their implementation uses private helpers.

`tools` is the chosen name for supported, ready-to-use compositions. They are neither throwaway examples nor a privileged internal agent layer. Tutorials and demonstration programs can still live in repository-level `examples/`, outside the public tool namespace.

### Dependency direction

- Representation implementations depend on shared operation contracts and optional integrations.
- Tools depend on public operations and, where appropriate, public tracing/training facilities.
- Operations never depend on tools.
- Training consumes traces and explicit component capabilities; tracing does not select an optimizer or training policy.
- Provider adapters do not own the public operation vocabulary.
- Importing `tensorcode`, an operation contract, or an LLM tool must not eagerly import unrelated tensor frameworks, models, or provider SDKs.

Keep the shared contracts lightweight. Install representation and integration dependencies through extras. Importing a package does not download weights, contact a provider, or create a global model.

## 4. Operation contract

The normal invocation is:

```python
output = operation(primary_input, context=context)
```

`context` is optional, keyword-only, and read-only by convention. An operation implements `forward`; users invoke the instance so framework hooks, tracing, and validation remain active.

Conceptually:

```python
class Operation(Generic[Input, Output, Representation]):
    def __call__(
        self,
        value: Input,
        *,
        context: Mapping[str, Representation] | None = None,
    ) -> Output:
        ...  # tracing/hook boundary around forward

    def forward(
        self,
        value: Input,
        *,
        context: Mapping[str, Representation] | None = None,
    ) -> Output:
        ...
```

This illustrates the contract, not a requirement to replace a tensor framework's native module hooks. A vector implementation should preserve native parameter registration, invocation hooks, device movement, and autograd behavior.

### Configuration versus invocation

Configuration holds stable choices: model/backend, shared backbone, output vocabulary, rubric, output type, and operation instructions. Invocation holds the current operand and conditioning context.

```python
assess_urgency = llm.Classify(
    model=model,
    labels=("routine", "soon", "immediate"),
    instructions="Assess the urgency expressed in the ticket.",
)

urgency = assess_urgency(message, context={"policy": encoded_policy})
```

The variable name expresses the operation's role. `assess_urgency`, `update_objective`, and `choose_handler` do not need corresponding globally named functions.

Required multi-part operands use a small typed input or an ordinary supported collection:

```python
choice = choose_handler(ChoiceInput(state=state, options=available_handlers))
updated = update_objective(RevisionInput(current=objective, evidence=observation))
combined = integrate((message, *pictures))
```

`ChoiceInput` is a candidate general operand contract. `RevisionInput` could be application-defined until repeated use justifies a shared type. Do not create a new framework dataclass for every use of a transform.

Context is conditioning data, not a hiding place for required operands, mutable global services, or arbitrary execution authority. Representation compatibility is checked for both primary operands and context.

### Primitive families

| Family | Contract | Important distinction |
|---|---|---|
| Encode | External data becomes a representation | Encoding may serialize, infer, or both; implementation states which |
| Decode | A representation becomes a requested external value | Decoding can lose information; it does not prove that value true |
| Transform | Construct or revise a representation | A generic transform is not evidence of general reasoning competence |
| Classify | Estimate membership in configured categories | Prediction about an input differs from choosing an action |
| Score | Assess an input on a specified scale/rubric | Relevance, utility, and probability are different quantities |
| Decide | Select or propose an option for an objective | Choosing an option does not execute it |
| Retrieve | Return existing items relevant to an input | Retrieval differs from inventing or transforming an item |

This is an initial vocabulary to validate with working programs. Comparison, ranking, checking, planning, and hypothesis formation may be compositions or specialized operations where their contracts warrant it. Do not make every cognitive verb a mandatory abstract method.

State-changing actions and persistent memory writes are explicit effects. They are not disguised as pure transforms.

## 5. Representation families

### Vector (`vec`)

Vector operations consume and produce tensors or tensor-backed representations. A `Latent` may carry shape, mask, space identity, and source references while preserving the underlying tensors and their gradients.

The representation contract includes the model/space identifier and version, dimensions, dtype/device expectations, and any token/spatial organization. Equal shape does not imply equal semantic space.

Image encoding must be able to retain spatial or relational organization; the interface must not require reducing every image to one undifferentiated vector. A text/image shared space must come from an actual compatible model or trained adapter, not a common wrapper class.

`Latent.zero()` is not a universal empty thought. If provided, construction requires a known space and shape, and the consuming model must define what that initial state means. Absence of state can instead be explicit.

### LLM messages (`llm`)

The representation is a structured message sequence with roles, content parts, and supported multimodal references. Operations can add, transform, query, or decode that representation through a supplied model or programmatic implementation.

A text encoder can simply construct a message; that is serialization, not a learned interpretation. A transform that calls a model has different behavior and cost. Document that distinction.

Do not flatten structured roles or image attachments into strings invisibly. Provider-specific serialization belongs in an integration.

### Symbolic graphs (`graph`)

The representation supports nodes, edges, attributes, and stable references with extensible labels. Source anchors, uncertainty, and alternatives can be attached without imposing one closed vocabulary of objects, intentions, scenes, or goals.

Graph operations may use deterministic algorithms, learned graph models, or external models. Symbolic representation does not imply rule-only implementation or differentiability.

The semantic graph is distinct from the execution DAG captured by tracing. One describes represented information; the other describes a computation's data dependencies.

### Crossing representations

Use explicit encoders, decoders, or adapters. They declare accepted spaces, information loss, and gradient support. Converting a graph to text and back is not assumed to preserve identity or meaning.

A family name identifies the representation exposed to the programmer, not the provider's hidden implementation. A hosted decision model such as Jev is not a `vec` implementation merely because it uses neural networks. It can be integrated through a suitable adapter, with conversions made explicit where necessary.

## 6. Results, uncertainty, and effects

Return ordinary usable values where sufficient. Classification and decision interfaces must also make distributions or scores accessible when the implementation supplies them, rather than always discarding them through argmax.

For example, a classification result can expose a selected value, per-label scores, their meaning, and an explicit abstention state. A result must not invent a probability or confidence value that its backend does not provide.

Keep distinct:

- An inferred answer and a decision to act on it.
- A missing prediction and a low-scoring prediction.
- Model probability, derived confidence statistic, relevance, and application utility.
- A predicted action outcome and an observed outcome.
- A failed operation and a valid abstention.

No universal confidence threshold is built into generic operations. Tools can supply documented policies, optionally fitted from validation data. Those policies remain replaceable.

Tracing metadata should normally live alongside results rather than forcing every tensor, message, or ordinary Python value into a heavy universal result envelope. The identity problem this creates is addressed explicitly below.

Errors such as incompatible spaces, invalid configuration, unavailable dependencies, and provider failure use explicit exceptions or documented operational outcomes. They must not silently become semantic conclusions or trigger an unrelated model fallback.

## 7. Shared models and execution

Operation identity and parameter identity are separate. Multiple operations can reference one model or backbone:

```python
encode = vec.TextEncoder(backbone=backbone)
update_objective = vec.Transform(backbone=backbone, head=objective_head)
choose_handler = vec.Decide(backbone=backbone, head=decision_head)
```

Sharing an object does not by itself eliminate repeated computation. Explicit shared encodings, backend batch APIs, or a compatible execution context provide that optimization. Do not promise automatic call fusion merely because operations share a model.

The trainer deduplicates shared parameters by identity. Checkpointing preserves shared references instead of restoring separate copies accidentally.

Start with eager execution. Independent operations may use ordinary concurrency or backend batching; dependent calls preserve their order. Speculative execution applies only to declared effect-free work. Never speculate real-world actions merely because the corresponding decision computations can run in parallel.

Async provider calls need an explicit async interface, such as `await operation.acall(...)`, rather than a call that unpredictably returns either a value or an awaitable. Sync and async paths must preserve the same semantic and tracing contracts. Exact batch APIs remain an implementation design choice.

## 8. Tools: ready-to-use compositions

Tools offer complete capabilities built solely from public operations. They may contain state, application-specific schemas, domain policies, and ordinary Python control flow.

Two construction modes serve different users:

```python
# Convenient: assembles a documented set of compatible components.
router = SupportRouter(model=model)

# Explicit: caller supplies the composition's components.
router = SupportRouter(
    encode=encode,
    classify=classify,
    assess_urgency=assess_urgency,
    route=route,
)
```

These are alternative modes. Invalid or ambiguous mixtures should fail clearly. A supplied `model` must meet the constructor's documented capability requirements; accepting an arbitrary model object does not guarantee compatibility.

Tools should expose their configured components for inspection and replacement through documented APIs. “Opaque” means usable without studying the internals, not inaccessible or dependent on private machinery.

Agent tools can own objective, conversation, and memory state. Each instance owns its session unless the caller explicitly supplies shared storage. Concurrent turns on a stateful agent require a defined ordering or conflict policy.

A chatbot's HTTP server, upload handling, and frontend are interfaces around the tool. They are not dependencies of the cognitive operations or mandatory parts of the chatbot class.

## 9. Context-managed tracing

```python
with tc.trace() as episode:
    state = encode(ticket)
    assessment = assess(state)
    decision = decide(assessment)

episode.supervise(decision, expected_handler)
trainer.step(episode)
```

Tracing captures operations participating in the session. Each invocation records enough information to identify:

- The operation instance, implementation/configuration version, and parameter references.
- Ordered input/output ports, including conditioning context.
- Producer-consumer edges and external inputs.
- Start/completion/failure status and parent call scope.
- Relevant model, randomness, effect, and storage references.
- Feedback and explicitly selected training targets when subsequently attached.

Data can be retained inline, through artifact references, or only for the lifetime of an in-memory graph. Tracing must not silently persist every prompt, image, or credential. The caller chooses retention/storage policy; ephemeral tracing is useful on its own.

### What the context manager can actually observe

It observes TensorCode operation boundaries and explicitly instrumented integrations. It does not magically observe all Python function calls or reconstruct arbitrary arithmetic, string slicing, container mutation, or external side effects.

There are two related graphs:

1. **Operation execution DAG:** provenance between captured invocation ports.
2. **Native differentiation graph:** tensor computations recorded by a backend such as autograd.

The native graph may contain many computations between operation boundaries. The operation DAG is not a replacement for it.

Loops become distinct invocation nodes as they execute. A dependency always points to an earlier produced value/version. Mutable state is represented through successive versions or explicit read/write events; a single mutable node must not create cycles or overwrite earlier evidence.

### Value identity and lineage

Never infer a dependency merely because values compare equal or share a content hash. Two equal strings may have unrelated origins, and Python scalar identity can be ambiguous.

Representation objects can carry lightweight lineage handles. For primitive outputs or ambiguous aliases, the tracing API must expose explicit port handles. The concise `episode.supervise(decision, ...)` form is only valid when that value resolves unambiguously; otherwise the caller supplies the producing invocation/output handle.

Containers require path-aware lineage. Aliases and mutations require versioning or a snapshot policy. An uninstrumented transformation is marked as an opaque boundary with captured input/output as available; the tracer must not invent a differentiable edge.

### Session boundaries and concurrency

Use execution-local scope rather than process-global mutable state. Nested trace scopes have explicit parent relationships. Child async work can attach to the active session when context is propagated; cross-thread/process propagation must be explicit and documented.

An input produced by another session is a boundary input with an optional provenance link. Cross-session provenance does not automatically retain a gradient graph across all previous sessions.

## 10. From traces to training DAGs

The key optimization is to avoid storing an internal prediction as though it were an independent training input.

For:

```text
ticket → encode → latent → assess → assessment → decide → decision
```

the end-to-end example's external input is `ticket`; its target might be a corrected handler for `decision`. The latent and assessment remain connected intermediates.

**Drop redundant intermediate payloads, not the edges or the means to reconstruct them.** Training a decider on a detached saved assessment is a valid head-only training task, but it does not train the upstream encoder or assessor.

### Building a training view

1. Choose target output ports and attach explicit supervision, rewards, or constraints.
2. Traverse their captured dependencies to find required ancestors.
3. Identify external inputs, context, state snapshots, and nondifferentiable boundaries.
4. Retain the required operation/configuration identities and dependency edges.
5. Choose which payloads to retain, recompute, or treat as fixed boundary data.
6. Validate that the requested learning method is supported on the selected path.

The training view can prune unrelated calls while the original trace remains available according to retention policy. Terminal outputs are not automatically targets; users may supervise intermediate outputs too.

Supervision is external ground truth or an explicitly identified training signal. Saving a model's own answer as its target is self-distillation, not evidence that the answer was correct. Training examples must retain that distinction.

### Immediate gradient training

Retain the live backend graph, form a loss from explicitly differentiable outputs, backpropagate, and update the selected shared parameters. Do not detach, serialize, or convert tensors to Python scalars on this path unintentionally.

Ordinary backpropagation still needs saved activations or a recomputation strategy. The context manager does not make their memory cost disappear.

### Deferred training

Persist external inputs, targets, required state/configuration references, and the reconstructible program/operation dependencies. Recompute a differentiable forward pass with the intended trainable model, then optimize.

Record collection-time versions separately from training-time versions. Reproducing an old output and learning with updated parameters are different tasks.

Old traces may not uniquely reconstruct arbitrary Python control flow. Distinguish:

- Training along the recorded path, with recorded branch choices.
- Rerunning the program, which may choose new branches under updated parameters.

Neither mode differentiates through an ordinary discrete branch choice automatically. Exact replay additionally requires the relevant randomness, model versions, and external responses.

### Gradient and effect boundaries

| Boundary | Supported interpretation |
|---|---|
| Compatible differentiable tensor operations | Native backpropagation |
| Frozen encoder or recorded remote response | Fixed features; downstream training only |
| Discrete choice / sampled action | Explicit estimator, relaxation, supervised target, or policy-learning method |
| Symbolic transform | No gradient unless an implementation explicitly provides one |
| Remote model without training access | Observation/teacher signal or fixed boundary, not local SGD into the service |
| External action | Recorded observation; replay must not re-execute the effect by default |

Distillation, policy learning, prompt optimization, and graph-rule induction may later use traces. They are separate learning methods, not synonyms for differentiating a log. Unsupported requests must fail explicitly.

## 11. Worked example: support decisions

The tool assembles an encoder and two configured classifiers. Its routing policy is application-owned.

```python
from tensorcode.ops import llm
from tensorcode.tools.decision import SupportRouter

router = SupportRouter(
    encode=llm.TextEncoder(),
    classify=llm.Classify(
        model=model,
        labels=("question", "bug", "refund"),
        instructions="Classify the customer's main request.",
    ),
    assess_urgency=llm.Classify(
        model=model,
        labels=("routine", "soon", "immediate"),
        instructions="Assess the urgency expressed by the customer.",
    ),
    route=my_routing_policy,
)

with tc.trace() as episode:
    result = router(ticket)
```

Internally:

```python
def __call__(self, ticket):
    state = self.encode(ticket)
    intent = self.classify(state)
    urgency = self.assess_urgency(state)
    return self.route(intent, urgency)
```

`my_routing_policy` may use uncertainty to defer or choose a queue. It need not itself be a learned operation. If training requires its internal dependencies, it must be instrumented or treated as an opaque boundary.

A later correction can supervise the intent output directly using its trace port. End-to-end training from a queue correction requires a defined loss and supported path through the routing policy; do not assume a final string supplies gradients automatically.

This program needs no agent, conversation ledger, goal interpreter, or symbolic world model. A vector implementation can replace the components when compatible models and label contracts are supplied.

## 12. Worked example: multimodal chatbot

The ready-to-use surface:

```python
from tensorcode.tools.agents import Chatbot

bot = Chatbot(model=multimodal_model)

with tc.trace() as episode:
    reply = bot("What is happening here?", images=[image])
```

An explicit vector composition can implement a turn approximately as follows. Its components are configured operation instances, and memory access uses a separate store interface:

```python
def __call__(self, text, *, images=()):
    message = self.encode_text(text)
    pictures = tuple(self.encode_image(image) for image in images)
    observation = self.integrate((message, *pictures))

    objective = self.update_objective(
        (self.objective, observation),
        context=self.context,
    )

    candidates = self.memory.search(observation)
    relevant = self.select_memories(
        (observation, candidates),
        context={"objective": objective},
    )

    response = self.form_response(
        (observation, relevant),
        context={"objective": objective},
    )
    reply = self.decode_text(response)

    self.memory.append(observation, response)
    self.objective = objective
    return reply
```

Important assumptions:

- Text and image encoders produce compatible representations or use explicit adapters.
- The integration model actually supports their organization and modalities.
- Initial objective state and absent memory have defined representations.
- The memory store's search and mutation semantics are explicit and traceable when needed.
- A turn's state changes have a defined failure/commit policy; the sketch omits transaction handling.
- `update_objective` can preserve an existing objective. A separate change detector is optional when its result or compute savings are useful.
- Naming a transform `form_response` or `update_objective` does not supply the required learned capability.

The tool may retain source images, text, alternatives, and retrieval references for revision. A holistic scene model should account for layout and relations when answering a scene question. Merely feeding object labels to a decoder does not establish that capability.

## 13. Worked example: native gradient training

This sketch demonstrates a deliberately narrow differentiable path. `Classify` exposes tensor logits; the configured encoder and head are trainable and compatible.

```python
encode = vec.TextEncoder(backbone=backbone)
classify = vec.Classify(backbone=backbone, head=head, labels=labels)

for text, target_index in training_data:
    optimizer.zero_grad()
    with tc.trace() as episode:
        state = encode(text)
        prediction = classify(state)
        loss = cross_entropy(prediction.logits, target_index)

    loss.backward()
    optimizer.step()
```

The native tensor framework computes gradients. Tracing records the execution for inspection and possible later data construction. `cross_entropy` need not be a TensorCode operation for native autograd to work; exporting that loss as part of a replayable operation DAG would require a recorded loss specification or instrumentation.

A higher-level trainer can own the optimizer, target mapping, and loss construction. It should remain optional. A programmer using ordinary training code should not lose TensorCode's benefits.

## 14. Persistence, evaluation, and operational behavior

### Saving programs and experience

Separate model/tool checkpoints from trace datasets. Checkpoints include component configuration, parameters, compatible space identifiers, and shared-reference structure. Stateful tools additionally define what session state they save.

Trace persistence uses versioned records and explicit codecs for supported values, with artifact references for large inputs. An in-memory operation reference is insufficient for replay after restart. Arbitrary closures, unversioned code, or opaque resources can make a trace non-replayable; report that limitation rather than silently approximating it.

External credentials are configuration references, not serialized trace payloads. Retention and redaction policies must state when replay becomes impossible because necessary inputs were omitted.

### Evaluation principles

- Test operation contracts independently from model quality.
- Test a real composition, including context and representation conversion.
- Measure prediction quality separately from schema validity.
- Report authored policies, supplied models, supplied labels, and learned updates separately.
- Evaluate learning on held-out inputs and track regressions as well as improvement.
- Do not count self-authored graph fixtures as evidence of perception or language understanding.
- Measure batching/latency claims rather than inheriting a provider's advertised result.

### Required behavioral checks

1. Compatible operations compose; incompatible spaces fail clearly.
2. The same instance works with tracing enabled or disabled without changed semantic behavior.
3. Equal-valued independent outputs are not merged in provenance.
4. Context and mutable state dependencies remain represented correctly.
5. Exported examples retain external inputs and selected targets while preserving reconstructible intermediate edges.
6. A shared-parameter gradient path reaches the intended encoder/head exactly once per optimizer update.
7. Deferred training can recompute the supported path and exposes unsupported boundaries.
8. Replay never silently repeats an external effect.
9. A supported tool is implemented exclusively through public interfaces.

## 15. Replacement strategy and milestones

Jacob's requested reset is to preserve the old repository privately at `JacobFV/old-tensorcode-2026-09-20`, then delete the superseded implementation from the official repository rather than deprecate or relocate it there.

This document does not perform that archival or deletion. Before deletion, a separate execution task must verify that the private archive preserves the intended repository history, current files including relevant uncommitted/untracked work, and any explicitly included external artifacts. A remote mirror alone does not preserve untracked files or local model caches.

Work remains on `main` under the repository instructions. Preserve the owner's feedback and this new architecture series. Do not carry forward the old regression suite as a requirement to recreate the old APIs or semantics.

Proposed milestones:

1. **Public contracts and one usable path:** implement a small set of operations and one support decision tool using a supplied real backend. Prove ordinary Python usage without an agent runtime.
2. **Trace lineage:** capture compatible operation dependencies, explicit boundaries, context, and output targets. Verify identity and state handling before claiming arbitrary DAG reconstruction.
3. **Trainable vector path:** demonstrate live SGD and deferred training from retained external inputs, with held-out improvement and shared-parameter checks.
4. **Second representation:** exercise corresponding contracts through a different representation. Preserve meaningful differences instead of inventing fake parity.
5. **Stateful multimodal tool:** compose a chatbot with explicit memory, objective revision, and compatible image/text processing. Assess actual scene and conversational behavior.
6. **Broader graph and learning capabilities:** add only where concrete workloads justify the interfaces and corresponding measurements.

Each milestone states what works, which components and knowledge were supplied, what was learned, and what remains unsupported.

## 16. Settled decisions and remaining choices

### Settled in the design conversation

- Operations live under `tensorcode.ops.<representation>.<operation>`.
- Representation families are `vec`, `llm`, and `graph`.
- Supported high-level compositions live under `tensorcode.tools`.
- Operation objects follow a common callable convention; developers name their roles.
- Operations can share models and parameters.
- Context-managed tracing is independent of any agent harness.
- Internal values can be removed as standalone training inputs while retaining the dependencies needed for training.
- Differentiable execution, provenance logging, and learning from feedback are related but distinct capabilities.
- The replacement should not preserve the current implementation through compatibility machinery.

### Proposed details to validate during implementation planning

- Exact public ABC/protocol names and how they coexist with native tensor modules.
- The minimal reusable input/result types and how explicit trace-port handles are exposed.
- Native representations versus lightweight wrappers, especially for scalar identity and mutation.
- Trace storage format, recomputation policy, and initial supported codecs.
- Initial model integrations and which operations they genuinely implement.
- Async/batch interfaces and the first supported training methods.
- Which tools merit public support beyond the initial decision tool and chatbot.

These are bounded design questions, not reasons to build another universal runtime first.

## 17. Sources and motivation

- [Jacob's original feedback](0-jacob-valdez-feedback.md).
- The parent repository's original README and 2022 architecture notes: composable operations, multiple intermediate representations, and trainable programs. These are historical design intent, not proof of an existing implementation.
- [Decision-layer landscape](https://github.com/JacobFV/old-tensorcode-2026-09-20/blob/716056ba2cbe66402f28dc764c66bdc2195f1c09/docs/revival/19-decision-layer-landscape.md), read critically: schema-constrained APIs can express an explicit unknown alternative, and model confidence is not automatically calibrated probability of correctness.
- [TypeSafe introduction](https://docs.typesafe.ai/introduction): bounded questions composed in ordinary code.
- [TypeSafe use cases](https://docs.typesafe.ai/concepts/use-case-map): retrieval, routing, verification, feature extraction, and other application settings.
- [Speculative fan-out](https://docs.typesafe.ai/patterns/fan-out) and [composite scoring](https://docs.typesafe.ai/patterns/composite-scoring): independent judgments, shared execution opportunities, and caller-owned composition.
- [TypeSafe confidence](https://docs.typesafe.ai/confidence): distributions and derived confidence are separate interfaces to preserve accurately.

The defining test is whether a programmer can compose useful cognitive behavior, inspect its execution, and improve the supported parts from evidence without first adopting TensorCode's preferred agent architecture.

## 18. Self-critique rounds before the reset

### Round 1 — Does this repeat the architecture's original failure?

Risk: a comprehensive directory diagram and primitive table become a mandate to generate dozens of empty or domain-specific abstractions. Revision: the tree is a responsibility map; modules require working implementations, and cognitive roles remain developer-selected names on configured operations. The initial milestone must demonstrate composition before adding more families.

### Round 2 — Does tracing promise impossible learning?

Risk: object equality is mistaken for provenance; saved outputs are mistaken for external inputs; remote decisions are mistaken for differentiable nodes. Revision: use explicit output handles for ambiguous values, retain edges when dropping payloads, distinguish native autograd from the operation DAG, and reject unsupported replay/training requests. Supervision has an identified source; copying predictions is not new ground truth.

### Round 3 — Can someone actually use and verify the replacement?

Risk: only abstract protocols ship, or a provider-shaped fake makes the chatbot appear functional. Revision: require a runnable local real-data decision example, real parameter updates, and a callable tool. Supplied provider functions can demonstrate wiring in tests, but must be labeled fixtures. A live provider-backed chatbot is verified only when a real provider is configured. Record the implemented subset and remaining gaps separately from this vision document.

### Round 4 — Can the destructive change be recovered and understood?

Risk: deleting code before preserving dirty work, leaving stale CI and documentation, or claiming the old suite passed. Revision: checkpoint all current source work, record baseline test results, verify the private archive's commit and privacy, replace packaging/CI/documentation together with the code, and verify the new wheel in a clean environment. Keep historical measurements in the archive rather than importing old capability claims into the new README.
