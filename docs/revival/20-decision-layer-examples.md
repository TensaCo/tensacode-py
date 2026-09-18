# 20 — A decision layer in a normal request path

`examples/decisions/` is tensacode used the way ordinary software would use it: a back-office
service whose handlers ask typed questions and branch on the answers. No agent loop, no
planning, no text generation. Code owns control flow.

    python -m examples.decisions.service --port 8795 --tier cascade-scored --train data/banking77_train.csv

Then `http://127.0.0.1:8795/` for an operator UI, or use it as an API.

---

## 20.1 What it looks like from a backend

```python
from examples.decisions import decisions, tiers
from examples.decisions.audit import Audit
import tensacode as tc

runtime = tiers.runtime(tiers.cascade_scored(Path("data/banking77_train.csv"))[0])
audit = Audit()

with tc.use(runtime):                        # bind once per process, or per request
    result = decisions.handle(ticket, audit=audit)
```

`handle` is one pass of ordinary Python:

```python
t = triage(ticket)                            # one classify question, three derived answers
if t.refund_asked.status != "holds":
    return {"next": "no refund path"}         # Verdict, not a bool: 'unknown' is sayable
eligibility = refund_eligibility(ticket)      # a walk over policy data and charge records
if isinstance(eligibility, Unknown):
    return {"decision": "escalate"}           # 'I cannot tell' is not 'no'
money = gates()["money"].decide(eligibility.allowed, t.confidence)
```

Five decisions are exposed, each as a plain function and an HTTP route:

| Route | Question | Answer type |
|---|---|---|
| `POST /triage` | which department, how urgent, is money being asked for | `Department`, `Urgency`, `Verdict` |
| `POST /decide` | the whole handler including refund | dict with the gate's action |
| `POST /rerank` | order candidate passages | `[(Passage, Score(kind="relevance"))]` |
| `POST /supports` | does this passage back this claim | `Verdict` — holds / fails / **unknown** |
| `POST /replay` | re-decide stored state | the new decision, for drift |
| `GET /why?ticket=…` | why did you decide that | the claim chain, with sources |
| `GET /config`, `GET /trace` | thresholds and their basis; what answered | — |

---

## 20.2 How confidence gating is configured

A `Gate` turns an answer plus a `Score` into `auto` / `confirm` / `escalate`. Thresholds are
per consequence class, and ordered by what a mistake costs:

```python
DEFAULT_THRESHOLDS = {
    "routing": Thresholds("routing", auto=0.60, confirm=0.30),   # reversible
    "reply":   Thresholds("reply",   auto=0.80, confirm=0.50),   # a customer sees it
    "money":   Thresholds("money",   auto=0.95, confirm=0.70),   # money moves
}
```

Better, build them from measurement rather than taste:

```python
gate = Gate.from_measurement(
    "routing", curve,                  # (threshold, accuracy above it, n) on a validation split
    target_accuracy=0.95, confirm_at=0.30,
    basis="selective curve on the fit's own validation holdout, n=1003",
)
```

`from_measurement` picks the **lowest** threshold whose measured accuracy reaches the target,
and if none does it **disables auto entirely** rather than rounding down to the best
available. On our data it chose `auto=0.600`, and on the held-out test set the auto band came
out at **94.9% accuracy** against a 95% target — the threshold transferred.

Two behaviours worth knowing, both from `tensacode.outcomes`:

- **A gate refuses to threshold a score that is not a probability.** A relevance,
  similarity, uncalibrated or vote-share score returns `escalate` with
  *"not a calibrated probability: relevance"*. Thresholding a reranker score as if it were
  P(correct) is a normal way to ship a confident mistake; here it cannot compile past the gate.
- **`Unknown` escalates and is not zero.** An abstention never lands in a "low confidence but
  still act" band.

---

## 20.3 How to audit a decision

Every decision writes claims with evidence, so:

```
GET /why?ticket=T-2

ticket    T-2
intent    transaction_charged_twice   (source: impl:tfidf-logreg@1)
dept      billing                     (source: policy:department-map)
allowed   True                        (source: policy:refund)
clauses   R1
charges   C-1
· A charge billed twice may be refunded automatically up to £50. (C-1 duplicates C-0, £12.50)

claim chain:
  ticket:T-2 refund_allowed True
    ← observed in policy:refund via walk
```

The refund decision is deliberately **not** a model question: the policy is data
(`REFUND_POLICY`), the charges are records, and the answer is a walk over both, so every
refusal names the clause that refused it (R2 pending, R3 already reversed, R4 too old, R5
above the cap). `POST /replay` re-decides the stored input; under the same bindings the tiers
declare `deterministic=True`, so it must match, and under different bindings it shows what a
configuration change would have done to decisions already made.

---

## 20.4 Measured

`python -m eval.decisions.measure --data data --hotpot 300` →
`eval/results/decisions_measure.json`.

**Provenance, stated per row**, because a mapping we wrote is not evidence about the framework:

| Row | Whose labels |
|---|---|
| intent | **public label** (Banking77 test split, 3,080 rows); nothing of ours in the path |
| department / urgency / refund-asked | public text + public intent label + **our mapping** |
| rerank | **public label** (HotpotQA distractor gold supporting paragraphs) |
| citation support | **public label** (gold vs distractor paragraph) |
| refund eligibility | **not measured against public data** — policy and charges are ours; it is a deterministic walk, covered by unit tests |

### Intent decision, 3,080 Banking77 test rows

| Arm | Coverage | Accuracy (of attempted) | auto / confirm / escalate | Auto-band accuracy | ECE | p50 | Model calls |
|---|---|---|---|---|---|---|---|
| rules only | 0.056 | 0.9711 | 0 / 0 / **3080** | — | — | 0.03 ms | 0 |
| learned only | 0.938 | 0.9425 | 2843 / 46 / 191 | 0.9493 | 0.009 | 0.47 ms | 0 |
| cascade | 0.939 | 0.9412 | 2672 / 46 / **362** | 0.9465 | 0.010 | 0.48 ms | 0 |
| **cascade + scored rules** | 0.939 | 0.9412 | **2845** / 46 / **189** | 0.9480 | 0.011 | 0.48 ms | 0 |

Floors: majority 0.013, random 0.013 (77 classes). Cost: $0 per 1,000 decisions, basis
"declared 0.0 by every in-process implementation" — no tier here is metered, and the trace
says so rather than the number being assumed.

Derived rows, over attempted (our mapping, read with that caveat): department 0.984–0.986,
urgency 0.978–0.988, refund-asked 0.993–0.994. **Note that department accuracy is higher than
intent accuracy** — most intent confusions stay inside one department — so reporting only
"96% department accuracy" would flatter the system. That is why both are in the table.

### The finding that matters: an accurate tier with no confidence is worse than no tier

The keyword tier answers **173 of 3,080** messages at **97.1%** precision — better than the
learned tier's 94.3% — and reports no `Score`. So:

- **rules only**: 173 correct answers, and the gate escalates **all 3,080**, because an answer
  with no confidence cannot be gated.
- **cascade**: the rules answer first and *capture* those 173 from the learned tier, turning
  auto-decisions into escalations: **191 → 362**. The 171-case difference is exactly the rules'
  firings. A cheap tier that is more accurate made the system worse.
- **cascade + scored rules**: same rules, same order, but the tier now reports per-label
  precision measured on the fit's own validation holdout, as
  `Score(p, "probability", basis="banking77 train-holdout, n=1003; …")`. Escalations fall to
  **189**, slightly better than the learned tier alone (191), at the same accuracy.

The lesson generalizes beyond this example: **in a confidence-gated architecture every tier
must report a calibrated confidence, or it silently converts decisions into escalations.**
A rule that never fired on validation reports nothing and abstains (`unmeasured_rule`) rather
than claiming certainty.

And note what the cascade is *not* worth: even scored, it buys 2 escalations out of 3,080
against the learned tier alone, and costs a hair of accuracy (0.9412 vs 0.9425). This matches
[§13](13-schema-brittleness.md), where cascades had no headroom on open-domain QA — a
bounded business-classification regime does not rescue the cascade claim.

### Calibration (learned tier, 2,889 answers)

| Stated | Observed | n |
|---|---|---|
| 0.562 | 0.522 | 46 |
| 0.653 | 0.531 | 96 |
| 0.752 | 0.774 | 124 |
| 0.855 | 0.859 | 205 |
| 0.987 | 0.983 | 2418 |

**ECE 0.009.** The confirm band's own accuracy is 0.522 — the middle band really is the
uncertain one, which is the three-way gate's justification rather than a decoration.

### Rerank and citation support, HotpotQA distractor

| Metric | Value | Floor |
|---|---|---|
| recall@1 (gold supporting paragraph first) | **0.843** | 0.200 random |
| recall@2, either gold | 0.943 | — |
| both gold paragraphs in top 2 | 0.340 | — |
| MRR | 0.908 | — |
| p50 latency | 0.31 ms | — |

Citation support (100 questions, 1,000 paragraph judgements) with the deliberately weak
overlap checker: **holds on gold 0.43, holds on distractor 0.13**, and **72% unknown**. It
separates gold from distractor by about 3.3×, and spends most of its judgements in the unknown
band. That is the intended demonstration — the band is *used* rather than collapsed into
"false" — but as a citation checker it is weak, and a cross-encoder belongs here.

---

## 20.5 A gap in the library this example had to work around

**`tc.classify` drops the confidence.** `Output.score` is written by implementations —
`LinearTextClassifier` computes a temperature-scaled, threshold-calibrated probability — and
is then read by nothing: the facade returns `T | Unknown`, and the span does not record it.
A decision layer cannot gate on a confidence it cannot see, so
`decisions.classify_with_confidence` calls `Runtime.call` directly to keep it.

This is the highest-value fix for this use case. Either the facades should return the score
(e.g. an `Answer(value, score)` or a `classify.with_score` variant), or `call_many` should
attach it to the span. Everything in `gating.py` depends on having it.

Smaller notes:

- `Ref` requires a `kind:name` shape, so implementation names have to be wrapped
  (`impl:tfidf-logreg@1`) to be used as evidence sources.
- There is no built-in selective-accuracy curve helper; `eval/decisions/measure.py` computes
  one. `LinearTextClassifier.fit` already does the equivalent internally for its own
  threshold, so this logic exists twice.

---

## 20.6 Honest limitations

1. **The tiers are weak.** 94.3% on Banking77 from TF-IDF is a reasonable floor, not a
   competitive decision model. The framework's contribution is that swapping in a better one —
   including a decision-only model like Jev (§19) — is a change to `tiers.py` and nothing else.
2. **No coverage guarantee.** Thresholds come from a measured selective-accuracy curve on a
   validation split, not from conformal prediction. Under distribution shift they will drift
   with no alarm.
3. **The refund decision is not publicly measured.** The policy and charge records are ours;
   only its determinism and clause citations are tested.
4. **Department, urgency and refund-asked depend on our mapping**, and are reported separately
   for that reason.
5. **No serving engineering.** Single process, `http.server`, no batching across concurrent
   requests, no auth, no persistence — the audit store is in memory and lost on restart.
6. **The citation checker is a demonstration**, not a useful component.
7. **Latency figures are about TF-IDF**, not about tensacode.

## 20.7 Files

| Path | What |
|---|---|
| `examples/decisions/domain.py` | typed questions, policy as data, provenance notes on our mappings |
| `examples/decisions/gating.py` | `Gate`, `Thresholds`, `Gate.from_measurement` |
| `examples/decisions/decisions.py` | the five decisions; `classify_with_confidence` |
| `examples/decisions/tiers.py` | rules / learned / cascade bindings, BM25 reranker, `ScoredKeywords` |
| `examples/decisions/audit.py` | decisions as claims, `why`, `explain`, `replay` |
| `examples/decisions/service.py`, `ui.html` | HTTP API and operator UI |
| `eval/decisions/measure.py` | the measurement above |
| `eval/results/decisions_measure.json` | raw numbers |
| `tests/test_decisions.py` | 23 contract tests |
