# 19 — The decision-layer landscape, and where tensacode fits

Asked because we had no example of tensacode used the way ordinary software would use it:
querying, deciding, choosing inside a normal request path. This surveys the category that
has grown up around that shape, checks the specific product the question came with, and says
plainly where tensacode belongs and where it does not.

Everything below was checked in September 2026. Claims I could not verify are marked
**[unverified]** rather than repeated as fact.

---

## 19.1 Jev / TypeSafe AI — verified, with caveats

The survey that prompted this was substantially accurate.

**Confirmed** from the vendor's own documentation and several write-ups:

| Claim | Status |
|---|---|
| Decision-only model, no text generation | confirmed ([docs](https://docs.typesafe.ai/introduction)) |
| Three question types: **Choice**, **Score**, **Noul** (yes/no, returns 0–1) | confirmed; Choice and Score return `choice`/`score`, `probabilities`, `confidence` |
| Several questions in one call, evaluated in parallel and in isolation against one state | confirmed — the docs make the "no context-rot" argument explicitly |
| Calibrated confidence, and confidence is separate from the answer | confirmed; vendor calls the method **RLCD**, and for Choice/Score derives confidence from the shape of the distribution |
| 70–500 ms latency | confirmed as a vendor claim |
| Output tokens free; input $0.042/M | confirmed as a vendor claim ([pricing write-ups](https://www.developersdigest.tech/blog/typesafe-jev-system-one-models-release-guide-2026)) |
| Early access from 15 September 2026 | consistent with everything found |

**Not confirmed, and worth knowing:**

- **The headline speed/cost multiples are self-tested.** TypeSafe's own homepage figures
  (193.6× faster, 444.6× cheaper) come from four workflows built by its own model-capabilities
  team; the company acknowledges possible bias and says real-world gains likely sit lower
  ([report](https://ts2.tech/en/typesafe-ai-raises-40-million-for-jev-but-its-445x-cost-claim-is-still-self-tested/)).
  One independent test (Every) corroborated the *direction* — roughly 25× faster and 580×
  cheaper than a frontier model on extraction, 0.35 s vs 8.83 s per passage — but no
  large-scale independent reproduction has surfaced. **[partially verified]**
- **No published rate limits, token limits, or SLA.** The docs note "model jaggedness"
  for a specific version with known issues, which is honest but unquantified.
- Most of the *use cases* in the survey (FNOL, AML, CV screening, moderation, code review
  gates) trace to vendor cookbooks and workflow designs rather than to named production
  deployments. One named production user (Notra) appears in secondary reporting.
  **[unverified as production volume]**

**The pattern it names is the important part**, and it is not proprietary:

> code owns control flow → the model answers several bounded questions in parallel →
> confidence decides auto / confirm / escalate → a generative model is used only when text
> must be produced.

Their confidence-routing pattern is a three-tier threshold keyed to consequence — under 0.6
route to a human, 0.6–0.85 acceptable for low-stakes actions, above 0.85 required for
high-stakes ones — with the explicit principle that *"the answer tells you what; confidence
tells you whether to act"* ([pattern](https://docs.typesafe.ai/patterns/confidence-routing)).
That is exactly the shape of `examples/decisions/gating.py`.

---

## 19.2 The wider category

| Approach | Interface | How confidence is exposed | How actions are gated | Production failure modes |
|---|---|---|---|---|
| **Decision-only models** (Jev) | typed questions (choice / score / yes-no) against a state | first-class: calibrated probability + separate confidence | threshold per consequence class | vendor-reported benchmarks; jaggedness per version; single-vendor dependency |
| **Structured outputs / constrained decoding** (OpenAI strict, JSON-schema modes, function calling) | a schema; you get conforming JSON | **none** — schema conformance is not confidence | usually ungated, or hand-thresholded on a field the model wrote | the schema is guaranteed, the *content* is not; ports between vendors break silently; retries on parse failure are invisible in headline rates and double the bill on the failure tail ([1](https://futureagi.com/blog/evaluating-llm-structured-output-modes-2026/), [2](https://arxiv.org/pdf/2601.06151)) |
| **LLM routers** (Not Diamond, RouteLLM, vLLM semantic-router) | a prompt in, a model choice out | varies; semantic routers expose similarity, learned routers a predicted-win probability | pick a model, not an action | static rules need maintenance as the task mix shifts; misrouting on edge cases causes retries and escalations ([Red Hat](https://developers.redhat.com/articles/2025/05/20/llm-semantic-router-intelligent-request-routing), [survey](https://techstrong.ai/articles/llm-routers-have-become-a-service-category-of-their-own/)) |
| **Guardrail layers** (Llama Guard, NeMo Guardrails, Guardrails AI, LLM Guard) | input/output validators, policy flows | usually a category label, sometimes a score | block / allow / rewrite before the user sees it | jailbreak and dual-use bypasses; policy drift; latency added to every call ([OpenGuardrails](https://arxiv.org/pdf/2510.19169)) |
| **Rerankers** (cross-encoders, Cohere/Voyage-class) | query + candidates → ordered list | a relevance score, comparable only within one ranking | top-k cutoff | the score is routinely mistaken for a probability and thresholded as one |
| **Feature stores / real-time ML** (Tecton, Feast-style) | features in the request path, model served behind an endpoint | model probability, often calibrated | business thresholds, champion/challenger | train-serve skew; feature freshness; the threshold lives in application code and rots |
| **Decision engines / rules** (DMN, OPA, flag systems) | decision tables, policy language | none — rules are certain by construction | the rule *is* the gate | rules cannot say "I don't know"; combinatorial growth; probabilistic predicates inside policy languages create conflicts ([DSL analysis](https://arxiv.org/pdf/2603.18174)) |
| **Selective prediction / conformal** (research, some production) | a predictor plus an abstention option | risk–coverage curves; conformal sets with finite-sample coverage guarantees | abstain and defer when uncertain, with a stated error target | requires exchangeability; degrades under distribution shift; needs a calibration set ([survey](https://arxiv.org/pdf/2508.07556), [clinical triage](https://www.nature.com/articles/s41598-026-40637-w)) |

The literature agrees on the mechanism we should be using for gating: **conformal / selective
prediction gives a principled way to turn a confidence into an auto-vs-defer decision with a
stated error rate**, rather than a hand-picked 0.85. Cost-sensitive variants exist for exactly
the imbalanced, high-stakes, human-in-the-loop case that back-office decisions are
([benchmark](https://arxiv.org/html/2607.27143), [risk control](https://arxiv.org/pdf/2603.24704)).
We do not do this yet; `Gate.from_measurement` is a selective-accuracy threshold on a
validation split, which is the honest first step but not a coverage guarantee.

---

## 19.3 Where tensacode fits

**What it genuinely adds in this shape:**

1. **The caller writes against a *question*, not a vendor.** `tc.classify(text, Intent)` is
   satisfied by keyword rules, a TF-IDF classifier, or a model, decided by deployment
   configuration. In `examples/decisions/` the five decision functions are byte-identical
   across four arms; only `tiers.py` changes. No decision API offers this, because the
   substitutable thing is the API itself.
2. **`Unknown` is a value, not a low score.** An abstention cannot be mistaken for 0.0
   confidence, and it propagates: an unknown intent leaves department, urgency and refund
   status unknown rather than defaulting to `general`. Schema-constrained APIs have no way to
   express this — a schema always gets filled.
3. **`Score.kind` makes a category error into a type error.** A reranker's relevance score
   cannot be thresholded as a probability; the gate returns `escalate` with
   *"not a calibrated probability: relevance"*. This is the single most common way a decision
   layer ships a confident mistake, and here it is refused by construction.
4. **Provenance and replay come free.** Every decision is claims with evidence, so
   "why did you refund this?" is answered by naming the clause, the charge and the tier, and
   the stored input can be re-decided under different bindings.
5. **Cost and model calls are counted from traces**, not asserted. "0 model calls" in the
   measurement is read from span attempts.

**Where it does not fit, and should not be pitched:**

1. **It is not a model.** Jev's value is that *someone trained a calibrated decision model*.
   tensacode is the layer where such a model would be plugged in — the two are complements,
   and a `JevClassifier` implementing `op="classify"` with `Traits(locality="remote",
   egress=True)` is the obvious integration, gated by policy like any other remote tier.
2. **Our own tiers are weak.** The measured ceiling in `examples/decisions/` is a TF-IDF
   classifier at 94.3% on Banking77. A decision-only model is likely better, and the honest
   framing is that tensacode makes swapping to it a configuration change.
3. **No coverage guarantee.** We threshold on measured selective accuracy; conformal
   prediction offers a distribution-free guarantee we do not provide.
4. **No serving story.** No batching across concurrent requests, no p99 budget, no
   multi-tenant isolation, no admin UI for thresholds. The example runs on
   `http.server`.
5. **Latency is only competitive because our tiers are tiny.** 0.47 ms p50 for a TF-IDF
   classifier is not a claim about the framework; it is a claim about TF-IDF.

**One-line summary:** the category is converging on *typed, bounded, confidence-gated
questions inside ordinary control flow*. tensacode is a good **shape** for that — typed
questions, honest abstention, score kinds, provenance, swappable tiers — and has no
competitive **model**. Its role is the seam, not the judgment.

---

## 19.4 Sources

- [TypeSafe AI — Introduction](https://docs.typesafe.ai/introduction)
- [TypeSafe AI — docs index](https://docs.typesafe.ai/llms.txt)
- [TypeSafe AI — confidence routing pattern](https://docs.typesafe.ai/patterns/confidence-routing)
- [Developers Digest — Jev benchmarked and priced](https://www.developersdigest.tech/blog/typesafe-jev-system-one-models-release-guide-2026)
- [ts2.tech — $40M raise, 445× claim still self-tested](https://ts2.tech/en/typesafe-ai-raises-40-million-for-jev-but-its-445x-cost-claim-is-still-self-tested/)
- [Cherry Creek News — Jev's own eval scores against two other models' answers](https://thecherrycreeknews.com/typesafe-jev-system-one-model-claims-evals-independent-tests-cherry_creek/)
- [Evaluating LLM structured output modes (2026)](https://futureagi.com/blog/evaluating-llm-structured-output-modes-2026/)
- [PromptPort: a reliability layer for cross-model structured extraction](https://arxiv.org/pdf/2601.06151)
- [Red Hat — LLM semantic router](https://developers.redhat.com/articles/2025/05/20/llm-semantic-router-intelligent-request-routing)
- [Techstrong — LLM routers as a service category](https://techstrong.ai/articles/llm-routers-have-become-a-service-category-of-their-own/)
- [vllm-project/semantic-router](https://github.com/vllm-project/semantic-router)
- [OpenGuardrails](https://arxiv.org/pdf/2510.19169)
- [OpenAI cookbook — how to implement guardrails](https://developers.openai.com/cookbook/examples/how_to_use_guardrails)
- [Uncertainty-driven reliability: selective prediction and trustworthy deployment](https://arxiv.org/pdf/2508.07556)
- [Conformal selective prediction with general risk control](https://arxiv.org/pdf/2603.24704)
- [Cost-sensitive conformal prediction and human-in-the-loop abstention](https://arxiv.org/html/2607.27143)
- [Conformal selective prediction for clinical triage under distribution shift](https://www.nature.com/articles/s41598-026-40637-w)
- [Conflict-free policy languages for probabilistic ML predicates](https://arxiv.org/pdf/2603.18174)
