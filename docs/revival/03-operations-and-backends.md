# 3. Operation algebra and backend protocol

## 3.1 Six families, eight facades

The six families held up as a *semantic* classification. They are not exposed as
six generic dispatch functions. Programs call typed facades, and each facade belongs
to exactly one family and fixes a contract.

| Family | Meaning | Facades (prototype) | Output | Uncertainty | Failure |
| --- | --- | --- | --- | --- | --- |
| **infer** | Estimate something not directly recorded | `parse(source, T)`, `classify(x, Labels)`, `choose(options, objective=, constraints=)`, `rank(query, candidates)` | `T` · a label · one of the feasible options · `[(candidate, Score)]` | Implementations abstain below an empirically chosen threshold. Scores declare their kind. | `Unknown(reason, detail, candidates)` |
| **check** | Evaluate a claim, candidate, or constraint against evidence | `check(proposition, evidence=)`, `verify(receipt, observe=, expect=)` | `Verdict(holds \| fails \| unknown, reasons, evidence)` | Three-valued. `unknown` is never `fails`. | `Verdict("unknown")` |
| **query** | Retrieve what is recorded | `Store.claims`, `Store.match`, `Store.neighborhood`, `Store.conflicts` | records / bindings / `Subgraph` | none (exact) | empty result |
| **rewrite** | Propose, then separately commit, a change to structured state | `propose(state, goal=, base_revision=)` → `Patch`; `Store.apply(patch)` → `Commit` | inert `Patch`; committed `Commit` | proposals can abstain | `Unknown`; `StaleRevision`; atomic rejection |
| **invoke** | Execute a registered action | `invoke(action, executor=, key=)`, `plan_order(plan)`, `run_plan(...)` | `Receipt(applied \| rejected \| failed \| indeterminate)` | `indeterminate` is explicit | never raises for executor errors |
| **convert** | Change representation and report what was lost | `to_records` / `from_records`, `encode` / `decode`, `pack`, `dedupe` | converted value + `ConversionReport` or `Packed.dropped` | n/a | `EncodeError`; `Unknown("required_evidence_exceeds_budget")` |

**No `retrieve` facade.** In every example, "retrieve" split cleanly into an exact
query (`bank.find_account`, `store.neighborhood`) and a relevance ranking (`rank`). A
`retrieve` facade would be a synonym for one or the other. It should come back only
when an index-backed store (vector or inverted index) makes "rank over an index" a
distinct contract from "rank these candidates".

**No `run(task=...)`.** Every facade has a typed target, and the runtime validates
outputs against it.

## 3.2 Distinctions the code enforces

| Distinction | Where it is enforced | Test |
| --- | --- | --- |
| Classification estimates what is true; action selection also involves objectives and constraints | `classify` has no objective. `choose` requires one, checks constraints itself *before* any backend sees options, and rejects backend answers outside the feasible set. | `test_choose_enforces_constraints_itself`, `test_choose_rejects_backend_output_outside_feasible_set` |
| Similarity is not a calibrated probability | `Score.kind ∈ {probability, uncalibrated, similarity, relevance, utility, vote_share}`; `probability` requires a calibration `basis`. `rank` only accepts `relevance` or `similarity`. | `test_scores_declare_their_kind` |
| Unknown is not false | `Unknown.__bool__` and `Verdict.__bool__` raise `TypeError`. `check` returns three states. An undetermined constraint excludes an option instead of passing it. | `test_unknown_and_verdict_have_no_truth_value`, `test_undetermined_constraint_excludes_option`, knowledge demo |
| A generated plan is not an executed action | `Plan` is data. Only `plan_order()` produces `RunnablePlan`, and `run_plan` requires one. `invoke` refuses an unregistered action, and a write with no idempotency key. | `test_plan_must_be_authorized_and_respects_dependencies`, `test_invoke_rejects_before_executing` |
| A proposed change is not a committed change | `Patch` is inert; `Store.apply` is the only commit path. Patches are atomic and checked against `base_revision`. | `test_patch_is_inert_atomic_and_revision_checked` |
| A conversion is not necessarily lossless | `ConversionReport` (aliasing, opaque types, identity collisions, unlinkable cycles); `Packed.dropped` lists every excluded item with a reason | `test_object_graph_identity_and_cycles_round_trip`, `test_pack_never_drops_required_evidence` |
| An executor's receipt is not verification | `verify` consults a fresh observation. An `indeterminate` receipt can verify `holds`, and an `applied` receipt can verify `fails`. | `test_verify_uses_observation_not_the_receipt` |

## 3.3 Mapping the existing operations

| Legacy op (`tensacode/ops/base/`, `.old/`) | What it actually meant | Proposed | Status |
| --- | --- | --- | --- |
| `encode` | Many things: to text, to vector, to "latent" | `convert` to a *named* target: `to_records`, text rendering, or an embedding owned by an implementation | **split**; the generic latent is removed |
| `decode` | Text/latent → typed value | `parse(source, T)` | **survives as `parse`** |
| `decide` (bool) | Either "is this true?" or "should we?" | `check(proposition)` → `Verdict`, or `choose([a, b], objective=…)` | **split by meaning** |
| `choice` (old5; conditions → functions) | Branch selection with a threshold | `choose` with constraints, plus ordinary `match` | **survives as `choose`** |
| `select`, `locate` | Find a part of an object | Exact: path tuple / `Store.match`. Fuzzy: `rank` over the parts | **split**; `Locator` → `tuple[str \| int, ...]` |
| `similarity` | Graded closeness (implemented as `==`) | `Score(kind="similarity")` from a ranking implementation | **becomes a score kind**, not an op |
| `predict` | Next item in a sequence | `parse`/infer into a typed forecast type; the reward loop is removed | **convenience** once a forecasting workload exists |
| `modify`, `correct` | Change an object toward a goal | `propose(state, goal=) → Patch`, then `apply` | **survives as `propose`** |
| `filter` (old) | Keep items satisfying a condition | `[x for x, v in zip(xs, check.many(...)) if v.holds]`, i.e. plain Python over a batch op | **removed**; composition |
| `query` | Fuzzy search in an object/context | Exact query methods, or `rank` | **split**; search stubs removed |
| `query_or_create` | Retrieve or invent | none | **removed**: conflates finding with fabricating |
| `split` | Invent categories and bin items | `classify.many` over a declared label type, then group-by | **removed** |
| `blend` | Interpolate objects | none | **removed** (no contract) |
| `transform` | Anything | none | **removed** (no contract) |
| `convert` | Encode, decode, modify | `parse` or `to_records` | **absorbed** |
| `plan` | Produce a plan | `propose(goal) → Plan` (data), then `plan_order` | **deferred** until a planning workload exists; the `Plan` type is implemented |
| `program` / codegen | Produce code | `propose(...) → CodeArtifact` (rewrite family) | **deferred** |
| `exec` / `call` | Run generated code / fill arguments and call | `invoke(RunCode(...))` of a registered, sandboxed action; argument filling = `parse(context, ParamsModel)` + normal call | **deferred** (needs a sandbox executor) |
| `loop` | The engine decides when to stop | Ordinary loop with explicit limits; `check`/`choose` for stop decisions | **removed** |
| `encode_args`, `autofill_args` | Implicit conversion at call boundaries | Explicit `parse` | **removed** |

## 3.4 Contracts without ceremony

Facades return **domain values**. They wrap only where a wrapper *is* the semantics
(`Unknown`, `Verdict`, `Receipt`, `Patch`, `Packed`). Execution metadata (which
implementation answered, attempts, escalation reasons, timing, cost) goes to the
active `Trace`. The main program therefore reads like ordinary code:

```python
intent = tc.classify(request.text, Intent)          # Intent | Unknown
if isinstance(intent, tc.Unknown):
    return cases.needs_human(email, "intent unknown", intent)
```

Detail is still one call away (`runtime.trace.of("classify")[-1].attempts`) and
serializes to JSONL.

Each facade validates backend output against its contract:
- `parse`: an instance of `T`
- `classify`: a member of the label enum
- `choose`: one of the feasible options
- `rank`: pairs with a relevance or similarity `Score`
- `check`: a `Verdict`
- `propose`: a `Patch` on the right base revision

An output that fails validation is an `invalid` attempt. It is never cached and
never returned, and the cascade continues.

## 3.5 Backend protocol

The whole protocol ([`runtime.py`](../../src/tensacode/runtime.py)):

```python
class Implementation(Protocol):
    name: str; version: str; op: str
    traits: Traits      # hard, declared: locality, egress, deterministic, requires={"cuda", ...}
    profile: Profile    # measured, nullable: source, quality{}, latency p50/p95, usd_per_call, peak_memory_mb
    def accepts(self, request: Request) -> bool: ...                          # capability check for this contract
    def run(self, requests: Sequence[Request]) -> Sequence[Output | Failure]: ...   # always batched

Output(value | Unknown, score: Score | None, usd: float | None)   # abstention is an Output, not an exception
```

Adapters in the prototype:
- `FunctionImplementation` / `@tc.implementation`: one function
- `KeywordClassifier`, `UtilityChooser`, `BM25Ranker`, `StoreFactCheck`: rules and classic algorithms, standard library only
- `LinearTextClassifier`: TF-IDF + logistic regression with temperature scaling and a validation-chosen threshold; `learned` extra
- `ChatClassifier`: a local instruction model through transformers; `local-model` extra

A remote-model adapter is the same protocol with `Traits(locality="remote", egress=True)`.

**Policy** is configuration, outside the program:

```python
Policy(localities={"in_process", "local_service"}, allow_egress=False, available={"sklearn", "cuda"},
       max_attempts=3, deadline_ms=None, max_usd_per_call=None, order="declared" | "cheapest" | "fastest",
       cache=True, record_inputs="full" | "digest")
Budget(usd=None, attempts=None, seconds=None)       # shared across calls, e.g. one agent episode
with tc.use(tc.Runtime(bindings, policy=policy, budget=budget)):
    handle(email, bank, cases)
```

**Routing algorithm** (`Runtime.call_many`):
1. Keep implementations whose `accepts()` holds for every request.
2. Order them as declared (static cascade), or by a *known* profile value, with unknowns last.
3. For each implementation, in order:
   - **Skip, with a recorded reason**, if it violates a hard constraint (locality, egress, missing capability, per-call cost cap, *unknown* cost under a cap), the deadline has passed, or the shared budget refuses it.
   - **Serve cache hits** if the implementation is deterministic and caching is on. The key is `(name, version, request digest)`. Cached abstentions count as abstentions.
   - **Run the rest as one batch.** A crash counts as a failed attempt. An answer is validated. An abstention adds its candidates to the eventual `Unknown`.
   - **Charge the budget** with metered cost when available, otherwise profile cost. Unknown cost is counted in `Budget.unmetered_calls` and never summed as zero.
4. Only still-pending items move on to the next implementation (`test_batch_cascade_only_escalates_pending_items`).
5. Exhaustion returns `Unknown` whose `reason` is the last real attempt's reason and whose `detail` is the whole chain.

**Deliberately absent: automatic quality-based routing.** The Banking77 measurements
([05-evaluation.md](05-evaluation.md#61-classify-cascade-on-banking77)) show why a
general model must not be assumed better. Qwen3-8B answered 173 of the 189 items the
learned tier abstained on, and was right on **38.2%** (95% CI 31.2–45.6%). That is
*worse* than the learned tier's own discarded argmax on those items (46.6%).
Coverage rose from 93.9% to 99.5%, and selective accuracy fell from 94.1% to 91.0%.

The next step is therefore `Policy.min_quality`. It admits an implementation only
when its `Profile.quality` for *this contract and slice* comes from a named
evaluation artifact and clears the bar. Unmeasured quality stays inadmissible
instead of being assumed good.

**Confidence thresholds need an empirical basis, and they drift.** The learned
tier's threshold (p ≥ 0.541 after temperature scaling) was chosen on a 10% training
holdout to reach 95% selective accuracy. On the untouched test split it reached
**94.3%** (CI 93.4–95.1%). The target sits at the CI's upper edge. Thresholds must be
revalidated per deployment slice, and their `basis` string travels with every
`Score(kind="probability")`.

## 3.6 Traces and provenance

Each `Span` records:
- `op` and `target`
- the input, or only its digest (`record_inputs="digest"` for sensitive data)
- output and outcome
- ordered `Attempt`s: implementation, version, outcome (`answer`, `abstain`, `invalid`, `error`, `skipped`, `cache_hit`), reason, backend time, and cost with its basis (`metered`, `profile`, `unknown`)
- notes (for example, which options constraints excluded)
- total time and backend time, so TensaCode overhead is `total − backend`
- parent section, and labels (for example, the idempotency key)

**Preserving the path to distillation.**
- **Training pairs.** Every span where a cheap tier abstained and a later tier answered is a candidate pair `(input, answer, answering implementation@version)`. Once an outcome is verified, by `verify`, a human resolution in the case log, or a later claim, the pair is labeled.
- **Abstention candidates** carry the cheap tier's top guesses with typed scores, for error analysis.
- **Versions are part of the cache key and the trace**, so a crystallized replacement can be evaluated against exactly the spans it would replace.
- **Deferred:** a feedback record type, trace storage, and any training platform. A workload with enough verified escalations for one contract comes first.

**Measured overhead** (Banking77 run; per call, p50/p95):

| Workload | Overhead per call, p50 / p95 |
| --- | --- |
| Trivial single implementation | 0.008 / 0.010 ms |
| Three-tier cascade with two abstentions | 0.012 / 0.013 ms |
| Real rules → learned cascade | 0.019 / 0.021 ms on 0.40 / 0.53 ms of backend time |

Cost scales with input size, because the request is digested for caching and the
trace. `rank` over ~40 snippets adds about 0.16 ms (the difference between p50 total and p50 backend time in the HotpotQA run). Computing the
digest lazily, only when caching or `record_inputs="digest"` needs it, is on the
plan.
