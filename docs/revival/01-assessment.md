# 1. Where `tensacode-py` actually stands

Inspected 2026-09-16 at `TensaCo/tensacode-py@6387f54` (`main`, clean tree), with
`TensaCo/tensacode@develop` (`6f42aaf`) for context and `JacobFV/tensacode@main`
(`573f75d`) for experimental ideas. Everything below marked **verified** was
executed; **read** means established by reading source only.

Environment for verification: Python 3.12.13 (uv), aarch64, pydantic 2.5.0 (the
exact version `pyproject.toml` pins). Reproduction commands are in
[`eval/README.md`](../../eval/README.md).

## 1.1 Summary

| Category | Finding |
| --- | --- |
| **Working** | Nothing in the active package works end to end. Three leaf modules import: `internal/consts.py`, `internal/utils/language.py`, and `ops/text_llm/python_str.py`. `defaults.py` also imports, but it is empty. |
| **Partially working** | `internal/utils/locator.py` path get/set logic, once its import error is fixed. 80 of 88 doctest examples pass. Serialization is lossy (see 1.3). |
| **Sketch** (docstring promises, trivial or placeholder body) | `similarity`, `plan`, `program`, `transform`, `decode_atomic`, `query` search strategies, `encode`, and every `text_llm` op |
| **Broken** (cannot run as written) | `engine.py`, all 21 `ops/base/*`, all TCIR modules, `meta/*`, `utils/misc.py`, `utils/pydantic.py` |
| **Abandoned abstractions** | Latent types (`TextLatent` … `Anthropomorphic`), the context/scope log, `reward/train/eval`, `TaggedObject`, `Serializable`, the class-based `BaseOp`, and `.old/old1`–`old6` (247 files, 13,205 lines) |

Building a wheel succeeds: `uv build` produces 44 files. That proves nothing, because
the wheel contains the same non-importable modules. `pip install tensacode` also
fails: PyPI returned HTTP 404 for `/pypi/tensacode/json`, `/simple/tensacode/`,
`/pypi/tensacode-py/json`, and TestPyPI on 2026-09-16. The package has never been
published under those names.

## 1.2 Import status (verified)

Each module was imported in a clean venv (pydantic 2.5.0, plus networkx/loguru/numpy
where a module reached for them). **37 of 41** modules fail to import. The 4 that import are an empty file, a version constant, a `Literal` alias, and a 17-line string helper. Every
first failure is listed below. Fixing one usually exposes the next.

| First failure | Modules | Evidence |
| --- | --- | --- |
| `No module named 'tensacode.core'` | 22 ops modules (17 in `ops/base`, 5 in `ops/text_llm`), `meta/tagged_object.py` | `core/` was flattened away in `aa5739c` (2024-09-29), but imports were not updated |
| `cannot import name 'Annotated' from 'pydantic'` | `engine.py` | `engine.py:26`; also `utils/misc.py:465` |
| `No module named 'tensacode.internal.utils.functional'` | TCIR `nodes.py`, `parse.py`, `graph_merging.py`, `ops/base/decode.py` | deleted in `1159261` (2024-07-23) |
| circular self-import | `utils/misc.py`, `ops/base/locate.py` | `misc.py:12` imports from itself |
| `'str' is not a valid discriminated union variant` | `utils/locator.py` | `locator.py:19` `type: str = Field(discriminator=True)` |
| `cannot import name 'UUID4' from 'uuid'` | `meta/schema.py` | `schema.py:2` |
| `No module named 'tensacode.internal.latent'` | `meta/param_tags.py` | moved to `meta/latent.py` in `290a2bb`, but importers were not updated |
| `No module named 'tensacode.internal.param_tags'` | `ops/base/autofill_args.py`, `ops/base/encode_args.py` | moved to `meta/` in `290a2bb` |
| `No module named 'networkx'` (undeclared dependency) | `meta/latent.py` | |
| `cannot import name 'consts' from 'tensacode.internal.utils'` | `utils/pydantic.py` | `consts.py` lives in `internal/` |

Layered failures appear once the first layer is shimmed:
- `meta/latent.py` (with networkx installed): `Tensor["h","w","c"]` raises `TypeError`, because a non-generic `Union` is not subscriptable (`latent.py:11`).
- `tcir/nodes.py` fails in four more ways:
  - `@abstractmethod` is stacked on `@property`, which is not writable (`nodes.py:45`).
  - pydantic has no schema for `complex`.
  - The recursive `Tensor` union recurses forever.
  - `@cached_property` has no `.setter`.
- The historical `polymorphic` dispatcher that TCIR depends on (`git show ea8c23f:tensacode/internal/utils/functional.py`) raises `TypeError` on its first registration (`tuple[int, Callable, Callable](...)`).

**History check.** Six snapshots were tested the same way: 2023-11-21, 2024-04-22, 2024-07-16, 2024-07-23, 2024-07-29, and 2024-07-30. The best result was **8 of 34** modules importing, and the engine never imported. The earliest snapshots may have targeted an older Python/pydantic (for example, `dataclasses._DataclassT`), so their result is a weaker claim. No snapshot ever contained a `LICENSE` file or an executable test.

## 1.3 Verified defects behind the previous inspection's claims

**Engine registries are shared mutable class state.** `engine.py:160-176` declares
`_instance_ops: ClassVar[dict] = {}`, and `register_instance_op` appends to it
through `self`. The pattern was run verbatim under pydantic 2.5.0. An op registered
on one `TextEngine` *instance* is visible from an `ImageEngine` instance, and all
three (`TextEngine`, `ImageEngine`, `Engine`) share the same dict object. The context
stack cannot even be created: `_all_updates: list = Field(...)` (`engine.py:272`)
raises `NameError: Fields must not use names with leading underscores`.

**Registration and dispatch are inconsistent.** There are four registration
interfaces, and none matches another:
- The engine defines `register_class_op` / `register_instance_op` (`engine.py:163,179`). Nothing uses them.
- The ops use `Engine.register_op_on_class`, which does not exist: 14 times bare, 11 times called with `()`, and 4 times with `score_fn=`.
- The engine docstring describes a class-based `BaseOp.execute` with `register_op_class_for_all_class_instances`.
- `ARCHITECTURE.md` describes `register_op(name, latent_types, score_fn)`.

Dispatch scores every candidate through `call_with_appropriate_args`, and positional
arguments are silently misaligned. `trace_wrapper` passes `context_overrides` into the
`fn_name_override` slot (`engine.py:661-667`). `modify` is defined twice
(`engine.py:843, 959`).

**TCIR reconstruction does not work (verified with import-only shims).** Details are
in [02-representation.md](02-representation.md#23-measured-legacy-behavior).
- `python_value` returns a `dict`, not the original type. Serializing a list of two tickets yields `{"items":[{},{}]}`, because `SequenceNode.items: list[Node]` serializes as the empty base class.
- Deserializing through `Node` fails (the class is abstract).
- `merge_identical` raises `AttributeError`.
- A dataclass with a field named `name` or `type` crashes `parse_dataclass`, since metadata and payload share one keyword namespace (`parse.py:292-298`).
- A Pydantic model crashes, because `dataclasses.fields` is called on it (`parse.py:328`).
- Cycles hit `RecursionError`.
- Generators and files are consumed while parsing (`parse.py:792`).
- `FileNode.python_value` calls `open()` with the stored mode (`nodes.py:279`), so reconstruction could truncate a file.
- `FunctionNode.python_value` calls `types.FunctionType(**params, body=…)` (`nodes.py:200`), which is not a valid call.

**The locator has validation and mutation problems (verified after fixing the import).**
- 8 of 88 doctests fail. The main cause: `DotAccessStep` checks `hasattr` before mapping membership, so dictionary paths take the attribute branch and raise `AttributeError`.
- `create_missing` on a list index path builds `{0: 6}` instead of `[6]`.
- `TerminalLocator.set` is a no-op (`locator.py:43`, with the comment "not sure if this actually does anything").
- `CompositeLocator.steps: list[LocatorStep]` serializes steps as the base class. `model_dump()` produces `{'type': 'composite', 'steps': [{'type': 'dot'}, {'type': 'index'}]}`, losing `key` and `index`, and re-validating that dump raises `TypeError: Can't instantiate abstract class LocatorStep`.

**Operations are much narrower than their descriptions (read).**
- **`similarity`** promises graded text and image similarity (`0.9` for near-paraphrases). Its body is `return 1.0 if input_a == input_b else 0.0` (`similarity.py:48-51`).
- **`query`**:
  - The `beam`, `breadth`, and `depth` strategies call `engine.generate_successors`, `evaluate_node`, and `is_goal`, none of which exists (`query.py:67-90`).
  - `greedy` is `pass`, followed by a loop of `locate → encode → transform` over "latents".
- **`loop`** raises `StopIteration` inside a generator (`loop.py:88-95`). Since PEP 479 that becomes `RuntimeError`, so every op built on it (`modify`, `blend`, `split`, `decode_list`) would crash at its stopping condition.
- **`predict`** reads `engine.c_pred_reward_coef`, which does not exist.
- **`split`** calls `engine.create`, which does not exist.
- **`call`** filters kwargs with `k not in args`, comparing keys to values (`call.py:104`).
- **`decode_*`** are called with `type=`, but the facade parameter is `type_`.
- **`plan`, `program`, `transform`, `decode_atomic`** raise `NotImplementedError`.
- **`text_llm/*`** calls `engine.llm.generate`, and no engine has an `llm` attribute.
- **Placeholders:** `reward`, `train`, `eval`, and `save` are `@abstractmethod`s on a `BaseModel`, so the abstraction is not enforced. There is no neural backend, no learning loop, no planner, and no executor.

## 1.4 Packaging, dependencies, licensing

**Dependencies (verified).** `pyproject.toml` writes `langchain = { version = "^0.1.16",
extras = ["llm"] }`. In Poetry, that installs langchain *with langchain's own `llm`
extra*; it does not make langchain optional. `langchain`, `dspy-ai`, `ivy`, `jax`,
`jinja2`, and `inflect` are therefore all **base** dependencies. Resolving them with
`uv pip compile` (Python 3.12, aarch64) yields **109 packages**, including `jax`,
`ivy`, `litellm`, `sqlalchemy`, and `scipy`. uv also warns that none of those
packages has an `llm` or `nn` extra.

Other problems:
- `[tool.poetry.extras]` names `tensorcode.llm` and `tensacode.nn`, which are not packages.
- `pydantic` is pinned to exactly `2.5.0`.
- `networkx` and `numpy` are imported but not declared.

**Metadata conflicts (read).**
- **Python version:** the README badge says Python 3.8+, `pyproject.toml` says `^3.11`, and `.python-version` says 3.12.
- **License:** the README and `pyproject.toml` say MIT, but no `LICENSE` file exists in any commit of `tensacode-py` or in `TensaCo/tensacode@develop`. Until one is added, "open source" is a claim without a grant.
- **Dead links:** the README links `CONTRIBUTING.md` and `CODE_OF_CONDUCT.md`, which do not exist, and gives install and import instructions (`from tensacode.core.base.base_engine import Engine`) that cannot work.
- **Stale layout docs:** the code-organization tree in both `README.md` and `ARCHITECTURE.md` describes `core/engine.py`, `operations.py`, `latent_types.py`, and `tests/test_engine.py`, none of which exist.
- **No CI:** the parent repo's GitHub workflow has no `on:` or `jobs:` keys.

**Examples and tests (read).**
- `examples/*.py`: five files with 0–1 lines each.
- `examples/{tensacode-agent,tensacode-operating-system,tensagame,tensaware,prompt-optimization}`: project skeletons. The only non-empty code is 48 lines of abstract agent classes and an 11-line optimizer stub.
- Tests: none. The five `tests/__init__.py` files are empty.

**History (read).** 92 commits from 2023-11-21 to 2024-11-23; 85 are titled "save" or similar. The architecture was reorganized four times:
1. **Engine mixins per backend** (`base/`, `llm/`, `nn/`; 2023-11 to 2024-04; now `.old/old5`).
2. **`core/base` with class-based ops and a two-part TCIR** (`tcir/data.py` plus `tcir/programming/*`, i.e. values and statements; 2024-07-16 to 07-22).
3. **Function ops** ("before switch to functional architecture" `5d15c3b`, "finish initial draft of basic ops" `d02a4a9`).
4. **Flattening** to `engine.py` plus `ops/` (`aa5739c`, 2024-09-29).

Each move copied docstrings forward and left imports behind.

## 1.5 Sibling repositories

**`TensaCo/tensacode@develop`** (parent monorepo). It contains no Python code of its
own. Language submodules (`typescript`, `csharp`, `java`, `cpp`, `dart`, `kotlin`,
`swift`) hold 3–4 placeholder files each. Its planning notes carry the most coherent
statement of intent:
- `planning/meetings/2023-08-02-ARCHITECTURE.md`
- `2022-11-21-planning.md`
- `2022-12-21-llm-functions.md`

That intent covers encode/decode/query over arbitrary objects, dispatch on Python
type × representation, runtime feedback, exporting learned subgraphs, and the
explicit caveat *"If you can write a complete and precise set of rules for your
problem, you're probably better off with traditional software engineering."*

**`JacobFV/tensacode@main`**: 2 commits on 2025-05-10 over about 5 hours, 6 Python
files, no tests, no license. A separate agent verified it by execution:
- **`BaseEngine`:** every method is `pass`.
- **`NNEngine`:** real torch code for number/string encoders, a learned soft `_if`, and select→`glom.assign` updates. It was not run (it requires torch), and `_decode_list` calls an undefined `_decode_new_item_model`.
- **`LLMEngine`:** prompts plain `gpt2` for JSON.
- **`tensacode_agents/agent.py`:** pseudocode that calls about 10 undefined methods and cannot be constructed.

Ideas worth keeping from it:
1. A model-gated `_if` as a *backend* for a typed check, rather than a control-flow primitive.
2. Locate-by-similarity followed by an explicit path assignment as one `rewrite` implementation.

It has no IR, tracing, or routing ideas to carry forward.

## 1.6 Keep / simplify / replace / remove / defer

Paths are relative to `tensacode-py`. "Migration" is what a user or contributor of
the current code must do. Because nothing imports today, no *working* user code can
break. The cost falls on docs, links, and contributors' mental models.

### Keep (the idea, re-expressed with less machinery)

| What | Source | Evidence it earns its place | Becomes | Migration |
| --- | --- | --- | --- | --- |
| Typed Python objects participate directly | README examples; `ops/base/decode.py` `decode_composite` | Every example program here uses dataclasses/enums as inputs and outputs | Facades take and return domain types (`parse(email, SupportRequest)`) | None |
| Meaning separate from implementation | `ARCHITECTURE.md` "operations specify latent types" and the dispatch idea | Same `agent.py` ran under rules+learned and rules+learned+Qwen3-8B | `Request` contract + `Implementation` protocol + `Policy` | Engine subclasses become implementations |
| Structured introspection / tracing | `engine.py` `trace_execution`, context log | Traces explain every Unknown and escalation in the examples | `Trace`/`Span`/`Attempt` (runtime.py) | Replace `engine.context` reads with trace queries |
| Feedback for later learning | `engine.feedback/reward` | Needed to turn escalations into training pairs | Spans with input digests, answering implementation, escalation reasons; feedback records deferred | None yet |
| Path access into nested values | `internal/utils/locator.py` | Rewrites need addressed edits | `SetField(ref, path: tuple[str\|int,…], value)` with copy-on-write (records.py `_replace_path`) | Locator objects become tuples |
| Differentiability / learned ops as a *possibility* | README, `nn` engines | Not demonstrated anywhere | An implementation may be a learned module; nothing in the core assumes tensors | Defer (below) |

### Simplify

| What | Source | Problem | Simplified form | Migration |
| --- | --- | --- | --- | --- |
| Engine | `engine.py` (967 lines) | Shared registries, 4 registration styles, context stack cannot be constructed, 20 thin wrappers | `Runtime(implementations, policy, budget)` bound with `tc.use(...)` (~390 lines incl. docstrings) | Do not subclass an engine; bind implementations |
| Op dispatch by score functions | `utils/misc.py` `score_inheritance_distance`, `engine.get_*_op` | Undefined ordering, no failure semantics | Declared cascade filtered by hard constraints; abstention and invalid output are explicit attempt outcomes | Score functions become `accepts()` plus policy order |
| TCIR | `internal/tcir/*` (1,973 lines) | Node per scalar, no identity, no provenance, unsafe reconstruction | Entities + claims over typed native values; see 02 | Parse nodes → `to_records`; nodes → entity payloads |
| Operation catalogue | `ops/base/*` (21 ops) | Synonyms and contract-free ops | 6 families, 8 facades; see 03 | See mapping table in 03 |

### Replace

| What | Source | Replacement | Why |
| --- | --- | --- | --- |
| Latent types | `meta/latent.py` (`TextLatent`, `ImageLatent`, `Anthropomorphic`, `Tensor["h","w","c"]`) | No core latent type. Representations are internal to implementations; `convert` names concrete targets | A universal latent type makes every op's semantics depend on an unspecified representation |
| `Encoded[T]` / `Autofilled[T]` and `encode_args` / `autofill_args` | `meta/param_tags.py`, `ops/base/{encode,autofill}_args.py` | Explicit `parse` into a parameter type, then an ordinary call | Implicit encoding hides cost, failure, and provenance |
| `loop` op (engine decides when to stop) | `ops/base/loop.py` | Ordinary loops with explicit limits; `check` or `choose` for stop decisions | Bounded, testable termination |
| Pydantic-everywhere internal models | `Engine(BaseModel)`, `Node(BaseModel)`, `Locator(BaseModel)` | Frozen dataclasses in the core; Pydantic *supported* as a user type | Pydantic caused 5 of the verified failures and forced a pinned version |

### Remove from the active package (history stays in Git)

| What | Source | Rationale |
| --- | --- | --- |
| Archived generations | `.old/` (247 files, 13,205 lines) | Unimported, and superseded four times |
| Empty examples and skeleton projects | `examples/*.py`; `examples/{tensacode-agent,…}` | No content; they suggest capabilities that do not exist |
| Contract-free ops | `blend`, `transform`, `query_or_create`, `split`, `correct` | No definable success criterion (see 03) |
| Sketch LLM ops | `ops/text_llm/*` | Depend on a nonexistent `engine.llm`; superseded by a typed implementation adapter |
| Unsafe reconstruction | `FileNode`, `ClassNode`, `FunctionNode.python_value` | Opens files, builds classes and functions from data |
| Misleading docs | README examples with invented outputs, the `ARCHITECTURE.md` layout, the MIT badge without a license | Replace with runnable examples and real results |
| Base-install heavy deps | `langchain`, `dspy-ai`, `ivy`, `jax`, `inflect`, `jinja2`, `glom`, `python-box`, `typingx`, `inspect-mate-pp`, `python-hooks`, `stringcase`, `attrs`, `loguru` | Core needs no third-party packages; backends declare extras |

### Defer (until a workload demands it)

| What | Condition to revisit |
| --- | --- |
| Differentiable control flow, `nn` engines, tensor latent graphs | A task where gradient flow through program structure beats training the leaf model separately |
| Compiling or crystallizing traces into learned operators | Enough traced escalations for one contract to train and *measure* a replacement (the trace schema already supports this) |
| Code generation plus execution (`program`, `exec`) | A sandboxed executor registered as an action |
| Persistent graph database | The reference `Store` stops fitting: it measured ~9k claims/s ingest and 106 MB traced memory at 112k claims |
| Other language ports | Python semantics stable; JSON trace and record schemas frozen |
| Repository consolidation (`tensacode` vs `tensacode-py` vs `JacobFV/tensacode`) | After the Python slice lands |
