# 2. Representation: coarser than TCIR, explicit where it matters

**Recommendation:** use typed native values by default. Add exactly two kinds of
graph record, **entities** and **claims**, only where identity, relationships,
evidence, time, or scope are needed. Keep executable structure (plans) in a separate
graph.

Criticism tested: *TCIR is too granular.* It is, and granularity is not its worst
problem. It has no identity, no evidence, and no reconstruction path. It also loses
data when serialized.

## 2.1 Method: start from the programs

The four example programs ([04-examples.md](04-examples.md)) were written first.
This table records what each concept in them actually required.

| Concept in a real program | Needs identity? | Needs explicit relationship? | Independently queried, updated, verified, or cited? | Verdict |
| --- | --- | --- | --- | --- |
| `SupportRequest.text`, `Card.last4`, `Printer.model` | no | no | no; read as a field | **plain typed value** |
| `Account.cards` (tuple of `Card`) | no; cards are addressed by `card.id` inside the value | no | only as part of the account | **nested value** |
| A printer, an account, a case | **yes**; referenced from many places, updated over time | from other records (`Printer.site: Ref`) | yes | **entity** |
| "PRN-3 was offline at 10:02" | yes; must be cited, contested, retracted | subject → object | yes | **claim** |
| "the monitor poll said so" / "the tech note said so" | the *source* needs identity | claim ← source | cited | source = **entity**; link = **evidence value** on the claim |
| Parser confidence, relevance score, utility | no | no | read alongside the value it scores | **`Score` value** with a declared kind |
| When the fact holds vs. when it was observed | no | no | yes (temporal queries) | **`Interval` on the claim**, `observed_at` on evidence |
| A hypothesis or "what the agent believes" | yes | claims are scoped to it | yes | **entity**, referenced by `Claim.scope` |
| Query variables (`?p located_on floor-2`) | no | n/a | only at query time | **`Var` in a query**, never stored |
| Constraints (`card_is_active`) | no | no | evaluated per option | **plain predicate** (`Constraint`), not a record |
| Steps of a plan and their ordering | yes (step ids) | **execution dependencies** | yes (receipts per step) | **program graph** (`Plan` / `Step`), separate from world claims |

No program needed a node for an integer, a string, a list, a type annotation, or a
function signature.

## 2.2 Alternatives compared

Worked example: two tickets that share one printer, which has a nested site. The
fixture is [`research/legacy_probe/fixtures.py`](../../research/legacy_probe/fixtures.py).

| | A. Legacy TCIR (fixed) | B. Native values only | C. Triples for every field | D. Property graph | **E. Values + entities + claims** |
| --- | --- | --- | --- | --- | --- |
| Records for 1 ticket / 2 tickets | 22 / 44 nodes *(measured)* | 0 / 0 | 17 / 24 triples *(hand-counted, incl. type triples)* | 3 / 4 nodes + 2 / 3 edges | **2 / 3 records** *(measured)* |
| Shared printer stored once | no *(measured)* | only as Python aliasing | yes (IRIs) | yes | **yes** *(measured)* |
| Two sources disagree about `status` | no place | no place (one field, one value) | needs reification per triple | needs promoting a property to a node | **two claims + evidence** |
| Validity time and scope | no | ad hoc fields | reification / named graphs | edge properties | **`Claim.valid`, `Claim.scope`** |
| Round trip to the original Python types | no *(measured)* | yes (pickle, unsafe) | custom | custom | **yes, registered types only** *(measured)* |
| Readability of one record | nested `{"value": …}` wrappers | best | poor | good | good (payload is the dataclass) |

**Why E.**
- **Versus B:** B cannot express the knowledge program. Two sources disagreeing about one field at overlapping times has nowhere to live.
- **Versus C:** C pays TCIR's granularity cost again, then pays more for provenance.
- **Versus D:** D is close, but evidence attaches to edges, and a property-level fact ("status") has to be promoted anyway. E makes that promotion explicit: a field is authoritative state, and a claim is a sourced assertion.
- **Versus A:** fixing A would still leave a node per scalar and no identity model.

## 2.3 Measured legacy behavior

This was produced by [`eval/representation_compare.py`](../../research/representation_compare.py).
The legacy half runs the committed TCIR code in a sandbox with six import-only shims,
each documented in [`research/legacy_probe/make_sandbox.py`](../../research/research/legacy_probe/make_sandbox.py).

| Probe | Legacy TCIR | Proposed records |
| --- | --- | --- |
| records, one ticket / two tickets sharing a printer | 22 / 44 | 2 / 3 |
| shared printer is one record | false | true |
| JSON of one ticket | 512 bytes; the `tags` strings are lost (`"tags": {"items": [{}, {}]}`) | 427 bytes, lossless |
| JSON of two tickets | `{"items":[{},{}]}` (everything lost) | three readable records (below) |
| reconstruct | `dict`, not equal to the original | `Ticket`, equal, and the shared printer is the *same object* |
| deserialize | `Node.model_validate_json` → `TypeError` (abstract class) | `Store.from_json` / `decode` |
| field named `name` | `TypeError: multiple values for keyword argument 'name'` | encodes normally |
| Pydantic model | `TypeError: must be called with a dataclass` | encodes and round-trips |
| cycle | `RecursionError` | through entities: restored; through values: refused with an actionable message |
| `merge_identical` | `AttributeError` | n/a: identity is explicit, never inferred from equality |
| unknown type name on load | n/a | `Opaque(...)`, nothing imported |
| update a nested field | mutate `node.printer.floor.value`; `python_value` then yields `{'value': 3}` | `Patch(SetField(ref, ("floor",), 3))` → new revision; the prior value is untouched |

## 2.4 The schema

```text
Ref(id="kind:name")                           stable reference; equality is identity

entity:   Ref ──► typed value                 (dataclass / Pydantic / enum / primitive; may contain Refs)
claim:    (subject: Ref, predicate: str, object: Ref | value,
           valid: Interval, scope: Ref | None)          content identity: claim:<sha256>
          └─ evidence: [Evidence(source: Ref, observed_at, locator?, method?, confidence: Score?)]
          └─ retracted: Retraction(reason, evidence)?

values:   Interval(start?, end?)   Evidence(...)   Score(value, kind, basis)   Opaque(type_name, data)
query:    Var(name)                                        (never stored)
schema:   Store.declare(predicate, functional=True|False)  (functional ⇒ contradictions are detectable)
program:  Plan(steps: [Step(id, action, needs: [step id])])   separate graph; see actions.py
```

**What shares a representation.** Observations, documents, sources, hypotheses,
contexts, cases, and devices are all *entities* whose values have domain types. The
schema does not grow a class per ontology concept. Evidence, intervals, and scores
are values attached to the thing they describe:
- Uncertainty about a *claim's source* goes in `Evidence.confidence`.
- Uncertainty in an *estimate* goes in the `Score` returned with it.
- Disagreement *between sources* is represented structurally, as two claims.

**Rule for promoting a field to a claim.** A field in an entity's value is the
system of record's current state and needs no citation. Once a fact can be
uncertain, time-bounded, sourced, contested, or scoped to a hypothesis, it becomes a
claim. In the knowledge example, `Printer.model` stays a field. Status is a claim.

**Identity.**

| What | How identity is set |
| --- | --- |
| Entities | *Assigned* identity (`Ref`). |
| Claims | *Content* identity: the same proposition from two sources is one claim with two pieces of evidence. |
| Observations | Entities with assigned, immutable ids. |
| Merging entities | Always an explicit rewrite. |

Two distinct objects that report the same identity key are *reported* as a
collision, not merged (`test_identity_collisions_are_reported_not_merged`).

**Contradiction** is a structural query, with no model involved. Two live claims
conflict when they share subject, predicate, and scope, the predicate is declared
functional, their objects differ, and their `valid` intervals overlap.

## 2.5 Python objects

| Concern | Behavior |
| --- | --- |
| Dataclasses | Fields enumerated with `dataclasses.fields`; rebuilt through `__init__`, with non-init fields set afterwards. |
| Pydantic | Fields from `model_fields`; rebuilt with `model_validate`, so validators run. Pydantic is supported, not required. |
| Registration | `TypeRegistry.register(cls, identity=...)` is an explicit allow-list. There is no global registry: every `Store` / `to_records` call takes one. |
| Unknown types | Encoding an unregistered object raises `EncodeError` (no pickling, no `repr` fallback). Decoding an unknown `$type` returns `Opaque(type_name, data)` and records it in the report. Nothing is imported by name. |
| Identity vs equality | Within one conversion, the same Python object reached twice is one record. Equal but distinct objects stay distinct. Graph identity comes from `Ref`, not from `==` or `id()`. |
| Aliasing | A mutable *value* (list, dict, non-entity dataclass) reached twice is duplicated in the output, and the loss is reported (`aliasing: Ticket:T-101.tags is the same object as Ticket:T-102.tags`). To preserve the sharing, register the type as an entity. |
| Cycles | Through entities: stored as `Ref`s and restored by two-phase reconstruction. Through values: refused with a message naming the path and the fix. Cycles through *frozen* dataclasses cannot be relinked after construction; that is reported as a loss, not faked. |
| Serialization | JSON with `$type`, `$ref`, `$enum`, `$tuple`, `$set`, `$map`, `$datetime`. Every lossy step (aliasing, opaque types, identity collisions, unlinkable frozen cycles) goes into a `ConversionReport` returned beside the data. |

## 2.6 World graph vs program graph

They may share a substrate, but their edges mean different things:

```python
# world: a relationship between facts (no ordering implied)
world.tell(tc.Claim(prn3, "located_on", floor2), tc.Evidence(asset_db_export, t))

# program: an execution dependency between effects (no truth claim implied)
# (same API as tests/test_actions_context.py::test_plan_structure_is_checked_and_dependencies_respected)
plan = tc.Plan((tc.Step("freeze", FreezeCard("card-a1")),
                tc.Step("notify", SendArticle(email, "kb/lost-card"), needs=("freeze",))))
runnable = tc.plan_order(plan)              # a Plan is data; plan_order only checks its structure
receipts = tc.run_plan(runnable, executor=bank, key_prefix="m-001")
```

A claim edge never schedules anything. A `Step.needs` edge never asserts anything
about the world. `run_plan` runs a step only if every step it needs came back
`applied` (`test_plan_structure_is_checked_and_dependencies_respected`).

## 2.7 Worked example, before and after

**Before (legacy TCIR, one ticket, measured).** Every scalar is a node, and metadata
shares a namespace with fields:

```json
{"type": {"value": "dataclass"}, "name": {"value": "Ticket"}, "id": {"value": "T-101"},
 "printer": {"type": {"value": "dataclass"}, "name": {"value": "Printer"},
             "asset_id": {"value": "PRN-3"}, "model": {"value": "LaserJet M507"},
             "site": {"type": {"value": "dataclass"}, "name": {"value": "Site"},
                      "label": {"value": "HQ"}, "timezone": {"value": "America/Chicago"}},
             "floor": {"value": 2}, "status": {"value": "online"}},
 "reporter": {"value": "dana"}, "text": {"value": "P3 jams on every duplex job"},
 "priority": {"value": "normal"}, "tags": {"items": [{}, {}]}}
```

The second ticket repeats all of `printer` (44 nodes total). Serializing both yields
`{"items":[{},{}]}`.

**After (proposed, two tickets, measured):**

```json
{"Ticket:T-101":  {"$type": "Ticket",  "fields": {"id": "T-101", "printer": {"$ref": "Printer:PRN-3"},
                   "reporter": "dana", "text": "P3 jams on every duplex job", "priority": "normal",
                   "tags": ["hardware", "jam"]}},
 "Ticket:T-102":  {"$type": "Ticket",  "fields": {"id": "T-102", "printer": {"$ref": "Printer:PRN-3"},
                   "reporter": "lee", "text": "P3 shows offline from floor 2 laptops", "priority": "high",
                   "tags": ["network"]}},
 "Printer:PRN-3": {"$type": "Printer", "fields": {"asset_id": "PRN-3", "model": "LaserJet M507",
                   "site": {"$type": "Site", "fields": {"label": "HQ", "timezone": "America/Chicago"}},
                   "floor": 2, "status": "online"}}}
```

Three records. `Site` stays a nested value because nothing refers to it
independently. `from_records` returns two `Ticket` objects equal to the originals,
and `rebuilt[0].printer is rebuilt[1].printer`.

**Manipulation:**

```python
patch = tc.Patch((tc.SetField(tc.Ref("Printer:PRN-3"), ("floor",), 3),), store.revision, "moved upstairs")
store.apply(patch)     # atomic; StaleRevision if the store moved on; prior value objects untouched
```

**When the status becomes contested**, it moves from the payload into claims (the
knowledge example, [output](../../examples/knowledge/OUTPUT.txt)). There
are two claims with evidence, one `Conflict` over 10:02–10:02, and
`status_at(10:02) → Unknown("contested")`. `check(online @10:02)` returns `unknown`,
not `false`. A retraction is a patch; the retracted claim is kept for audit.

## 2.8 Cost of the reference store (measured)

From [`eval/results/graph_bench.json`](../../eval/results/graph_bench.json).
Setup: 8 status claims per device from 3 sources, CPython 3.12, one Cortex-X925
core. The store is in-memory and single-threaded, so this bounds the prototype. It
is not a database benchmark.

| claims | ingest claims/s | `claims(subject, predicate, at)` p50 | `conflicts(subject)` p50 | 2-pattern `match` p50 | `SetField` patch p50 | traced memory | JSON save / load |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,125 | 9,981 | 0.002 ms | 0.009 ms | 0.033 ms | 0.003 ms | 1.2 MB | 0.02 s / 0.03 s |
| 11,250 | 9,534 | 0.004 ms | 0.010 ms | 0.038 ms | 0.003 ms | 10.8 MB | 0.21 s / 0.29 s |
| 112,500 | 8,737 | 0.006 ms | 0.013 ms | 0.076 ms | 0.003 ms | 106 MB | 2.25 s / 3.18 s |

Before a fix made during this work, commits copied the whole entity map, and patch
latency grew from 0.003 ms to 0.074 ms with store size. Staging only the touched
entities made it flat. Ingest is dominated by content hashing: SHA-256 over canonical
JSON, once per claim.

**Limitations:**
- `match` is a naive nested-loop join.
- Whole-store JSON is the only persistence.
- There is no index on predicate or object values other than `Ref`s.
- `records.py` is the largest module (612 non-blank lines); `to_records` / `from_records` could move out of the core if they are not needed.
