# 6. Evaluation: design, measured results, limitations

All numbers come from `eval/results/*.json`, produced by the scripts in
`eval/` on 2026-09-16. Nothing is mocked. The model tier is a real
Qwen3-8B running locally. Simulations are labeled as simulations.

**Hardware and software.**

| Component | Version / spec |
| --- | --- |
| Machine | NVIDIA GB10 (Grace-Blackwell SoC): 20-core Arm CPU (Cortex-X925/A725), 121.6 GB unified memory |
| OS / kernel | Linux 6.17.0-1026-nvidia |
| Python | CPython 3.12.13 |
| ML libraries | torch 2.13.0+cu130, transformers 5.17.0, scikit-learn 1.9.1, numpy 2.5.3 |
| Code | `tensacode-py@6387f54` plus the uncommitted the repository |

Latency figures are single-process and single-request unless stated otherwise.

## 6.0 Evaluation design

| Dimension | Metric | How it is kept honest |
| --- | --- | --- |
| Task quality | Accuracy; *selective* accuracy on answered items; coverage | Official held-out test split. No fitting, threshold choice, or rule writing touched it. Wilson 95% intervals. |
| Abstention / calibration | Coverage at a threshold chosen on validation; ECE (15 bins) before and after temperature scaling; accuracy of each escalation tier *on the items it received* | Validation = stratified 10% of train (seed 0). The target was fixed before test. The test miss is reported. |
| Latency | p50/p95/p99 per call: `total`, `backend`, and `total − backend` (TensaCode overhead) | Warm-up excluded; cache disabled during latency runs. Batched per-item times are marked as amortized. |
| Throughput | items/s, per item vs batched | Same inputs, and output equality between the two modes is asserted |
| Memory | Process RSS; CUDA peak allocated; `tracemalloc` for the store | |
| Cost | Metered USD (all local here: none); GPU-reported energy above idle | Energy is the nvidia-smi GPU rail sampled every 250 ms, minus a 3 s idle baseline. It excludes CPU, memory, and system power. |
| Structured graph transformations | Correctness (tests plus the legacy comparison); scaling of ingest, temporal query, conflicts, joins, patches, and save/load | Reference in-memory store; synthetic uniform data |
| Agent trajectories | Per episode against ground truth: done exactly once, duplicate effect, false done, escalated with/without effect; invocations; simulated time; CPU | Baseline policies run on identical sampled worlds; a mis-specified-facts sensitivity run is included |
| End to end | Demo traces include parsing, routing, caching, verification, retries | See 04 |

**Not measured** (see 6.6): a remote model tier, real traffic, concurrency, and
downstream answer quality.

## 6.1 `classify` cascade on Banking77

**Dataset.** Banking77 (PolyAI, CC-BY-4.0): 10,003 train and 3,080 test utterances,
77 intents.
- Train sha256 `b06e26ac…c664b`; test sha256 `d12d6e3b…b474d`.
- **Rules:** written from training data, then revised once against errors on the training holdout. v2 was frozen before test.
- **Learned tier:**
  - Model: TF-IDF (word 1–2-gram + char 2–5-gram) with multinomial logistic regression (C=20).
  - Training data: 9,000 examples; 16 s fit on the CPU.
  - Calibration: temperature scaling on the 1,003-example holdout (T=0.716).
  - Threshold: 0.541, the lowest value reaching 95% selective accuracy on the holdout (holdout coverage 93.7%).
- **Model tier:** Qwen3-8B, bf16, `enable_thinking=False`, greedy decoding. The zero-shot prompt `classify-v1` lists all 77 labels, and the model replies with one label or `unknown`.

| Binding | Coverage | Selective accuracy (95% CI) | Errors | Notes |
| --- | ---: | ---: | ---: | --- |
| rules alone | 5.6% (173) | **97.1%** (93.4–98.8) | 5 | 30 ms for all 3,080 |
| learned alone, no abstention | 100% | 91.3% (90.2–92.2) | 268 | ECE **0.017** after temperature scaling (0.068 before) |
| learned alone @ holdout threshold | 93.8% | **94.3%** (93.4–95.1) | 166 | Target was 95%; it sits at the upper edge of the CI. Revalidate thresholds per slice. |
| **rules → learned** (TensaCode cascade) | 93.9% | 94.1% (93.2–94.9) | 170 | Rules answered 173, learned 2,718, Unknown 189 |
| model alone (500-item random subset, seed 0) | 93.6% | **68.2%** (63.8–72.2) | 149 | 15 declined, 17 unparseable. The learned tier scored 91.2% on the same 500. |
| **rules → learned → model** | **99.5%** | **91.0%** (89.9–91.9) | 277 | 189 escalated; the model answered 173 and was correct on **38.2%** (31.2–45.6). The learned tier's discarded argmax on those items was 46.6%. |

**Latency, throughput, memory, energy:**

| Measure | Value |
| --- | --- |
| rules → learned, per call | total p50/p95/p99 = 0.42 / 0.55 / 0.68 ms; backend 0.40 / 0.53 / 0.66 ms; **TensaCode overhead 0.019 / 0.021 / 0.026 ms** |
| Throughput, per item vs batched | 2,294 items/s through TensaCode; 2,677 items/s calling the same two tiers directly; **14,827 items/s** batched (`classify.many`, chunks of 256); outputs identical |
| Runtime overhead, trivial implementations | 1 tier: 0.008 / 0.010 ms (p50/p95); 3 tiers with 2 abstentions: 0.012 / 0.013 ms |
| Memory | RSS 734 MB after fitting (training data included), 741 MB after the runtime runs |
| Model tier: load | 81 s cold (weights from local disk) |
| Model tier: latency | batch 16: 243 ms/item amortized; batch 1: p50 518 ms, p95 823 ms |
| Model tier: resources | CUDA peak 17.7 GB; GPU mean 62 W while generating (idle 10.2 W) |
| Model tier: energy above idle | 12.6 J/item on the subset; 2,545 J for the full-test escalation run (189 escalated items) |
| Metered USD | none (all tiers local) |

**Conclusions this data supports:**
1. **Zero-shot general-model escalation was harmful on this in-domain, fine-grained task.** A general model is not a safe default for "the cheap tier is unsure". Admission must be gated on measured quality for that slice ([03 §3.5](03-operations-and-backends.md#35-backend-protocol)).
2. **The rules tier did not earn its place on quality.** It was 97.1% precise but made 5 errors, while the learned tier alone had 4 fewer total errors. It saves about 0.4 ms on 5.6% of traffic. Keep a rules tier only where it is exact (identifiers, formats) or where it encodes policy.
3. **TensaCode's runtime cost is small next to any real backend:** ~19 µs per call, with batching that preserves results and runs 6.5× faster than per-item calls.

**Not supported by this data:**
- Any claim about few-shot, fine-tuned, or larger models.
- Any claim about non-English or non-banking text.
- Any claim about production traffic distributions.

## 6.2 Context selection on HotpotQA

**Setup.**
- **Data:** HotpotQA distractor validation, all 7,405 questions (CC-BY-SA-4.0), sha256 `c20b638c…f7c6`.
- **Candidates:** the ~40 sentences of the 10 given paragraphs, each prefixed with its title.
- **Target:** the gold supporting-fact sentences.
- **Costs:** measured with `approx_tokens` (word and punctuation count, not a model tokenizer). On 12,187 of these sentences it counted 0.82× as many tokens as the Qwen3 tokenizer, so a budget of 128 here is roughly 156 Qwen3 tokens.

| Strategy | Budget | Mean supporting-fact recall | All gold included | Gold fits budget |
| --- | ---: | ---: | ---: | ---: |
| document order | 64 | 8.8% | 0.2% | 32.1% |
| **BM25 → pack** | 64 | **41.3%** | 7.9% | 32.1% |
| document order | 128 | 13.8% | 1.3% | 91.6% |
| **BM25 → pack** | 128 | **59.3%** | 26.6% | 91.6% |
| BM25 → dedupe(0.5) → pack | 128 | 59.3% | 26.5% | 91.6% |
| document order | 256 | 24.2% | 5.0% | 99.9% |
| **BM25 → pack** | 256 | **74.4%** | 48.6% | 99.9% |

- **Required-evidence invariant:** a gold sentence was marked required at each budget. It was included in **21,888 of 21,888** feasible cases, and the result was `Unknown("required_evidence_exceeds_budget")` in **327 of 327** infeasible cases.
- **Duplicate injection:** a lightly edited copy of every gold sentence was added. At a budget of 128, BM25 alone spent 29.7 tokens per question on copies (recall 51.1%). With dedupe it spent 0.65 tokens (recall **56.6%**). HotpotQA itself has few near-duplicates, so dedupe changes nothing on the clean data.
- **Latency per question, p50/p95:**

  | Stage | p50 / p95 |
  | --- | --- |
  | `rank` total | 0.45 / 0.70 ms |
  | `rank` backend | 0.29 / 0.46 ms |
  | `dedupe` | **4.6 / 10.4 ms** (O(n²) shingle comparisons, the dominant cost) |
  | `pack` | 0.12 / 0.18 ms |

  The ~0.16 ms `rank` overhead comes from digesting the 40 candidates for cache and trace.

**Interpretation.** A lexical ranker leaves 26–59% of supporting evidence out at
these budgets, and HotpotQA is built so that bridge facts share few words with the
question. This is the measurement a better `rank` implementation must beat. Dedupe
needs a blocking step (for example, MinHash) before it scales past tens of items.

## 6.3 Recovery trajectories

**Simulation setup.** 5,000 sampled worlds (seed 0), shared by all policies.
- **Per-invocation behaviors:** ok 65% (5 of those points with replica lag), 503 15%, reply lost after commit 8%, reply lost before commit 7%, 403 5%.
- **World facts:** executor honors keys 50%; authoritative read 50%; ledger query reachable 85% per call; replica lag in non-authoritative worlds (whenever the sampled script contains a lagging success, otherwise with probability 30%).
- **Action:** non-idempotent `CreditAccount`.

| Policy | Done exactly once | **Duplicate effect** | False "done" | Escalated, effect happened | Escalated, no effect | Mean simulated time | CPU / episode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **TensaCode recovery agent** | 90.0% (89.1–90.8) | **0** (0–0.08%) | 0 | 1.2% | 8.8% | 1.10 s | 0.053 ms |
| same, agent *wrongly* told keys are honored | 90.8% | **0.92%** (0.7–1.2) | 0 | 1.1% | 7.2% | 1.21 s | 0.055 ms |
| naive retry on any failure (≤3) | 92.4% | **4.76%** (4.2–5.4) | 0 | 1.0% | 1.9% | 0.71 s | 0.007 ms |
| verify, then retry without safety analysis | 76.2% | **12.74%** (11.8–13.7) | 0 | 3.9% | 7.2% | 1.00 s | 0.013 ms |
| single attempt | 64.1% | 0 | 0 | 8.1% | 27.8% | 0 | 0.005 ms |

**Interpretation.**
- **The trade-off:** the agent gives up 2.4 points of completion against naive retry and escalates 10% of episodes, and in exchange it moves no money twice under a correctly described target system.
- **Naive verification is worse than naive retry.** Verify-then-retry duplicates *more*, because a lagging replica read looks like "not applied".
- **The safety comes from stated facts, not from TensaCode.** When those facts are wrong, duplicates return (0.92%). This checks the decision logic under a fault model its author also wrote. It is not evidence about any real payments system.

## 6.4 Representation and store

See [02 §2.3](02-representation.md#23-measured-legacy-behavior) for legacy TCIR vs.
proposed records on the same objects (22/44 nodes vs 2/3 records, lossy vs lossless,
crashes vs round-trips). See [02 §2.8](02-representation.md#28-cost-of-the-reference-store-measured)
for store scaling to 112,500 claims.

## 6.5 Tests

`tests`: **43 tests**, all passing in about 0.05 s with no third-party
dependencies beyond pytest. They cover:
- **Routing:** cascade, abstention reasons, invalid outputs, crashes, hard constraints, unknown cost under caps, budgets, per-item attempt limits, deterministic-only caching, batch escalation of pending items only, cost ordering with unknowns last.
- **`choose`:** constraint enforcement and rejection of infeasible outputs.
- **Truth values:** three-valued `check`; no truth value for `Unknown` or `Verdict`.
- **Records:** content identity, conflicts, temporal queries, retraction, joins, atomic and revisioned patches, codec safety, JSON round trip, identity, cycles, aliasing, collisions.
- **Actions:** plan structure, key requirements, indeterminate receipts, verification by observation, plan ordering and refusal reasons and dependencies.
- **Context:** packing and dedupe invariants.
- **Example programs:** smoke tests with deterministic bindings.

## 6.6 Limitations

- **Banking77.** The learned tier and rules were tuned on this dataset's training split, so this is in-distribution performance. The model prompt is a single zero-shot version, and no prompt search was done, deliberately: that would have required a validation protocol for the prompt.
- **Energy.** Only the GPU-reported rail was measured. The whole-system draw of the GB10 was not.
- **Remote models.** None was run, so metered USD cost and network latency are unmeasured.
- **Concurrency.** The runtime is synchronous. There was no concurrency or multi-tenant load.
- **Deadlines.** Checked only between attempts. A single slow backend call cannot be interrupted.
- **Simulated components.** The recovery and support-router environments are simulations.
- **HotpotQA.** The metric is evidence recall, not answer quality. The token counter undercounts Qwen3 tokens by about 18%.
- **Store benchmark.** Synthetic, uniform, single-threaded, in-memory.
- **Unexercised code paths.** The `propose` facade has contract tests but no example workload. `run_plan` is tested but not used by an example.

## 6.7 Reproduce

```bash
cd proposal
uv venv && uv pip install -e ".[dev,learned,local-model]" pyarrow psutil   # local-model extra only for 6.1's model rows
python -m pytest -q
python eval/banking77_cascade.py --data-dir DATA [--model Qwen/Qwen3-8B]     # DATA holds banking77_{train,test}.csv
python eval/context_hotpot.py --parquet DATA/hotpot_distractor_validation.parquet
python eval/recovery_trajectories.py --episodes 5000
python eval/graph_bench.py
python research/legacy_probe/make_sandbox.py SANDBOX && python eval/representation_compare.py --legacy-python PY_WITH_PYDANTIC_2_5 --sandbox SANDBOX
```

The data sources and checksums are listed in `eval/README.md`.
