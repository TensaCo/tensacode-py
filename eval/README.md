# Evaluation scripts

| Script | Measures | Result file |
| --- | --- | --- |
| `banking77_cascade.py` | `classify` quality, abstention, calibration, latency split (backend vs TensaCode), throughput, memory, GPU energy; optional local-model tier | `results/banking77.json` |
| `context_hotpot.py` | rank → dedupe → pack evidence recall under budgets; required-evidence invariant; duplicate injection; latency | `results/context_hotpot.json` |
| `recovery_trajectories.py` | Complete simulated recovery episodes against ground truth for 5 policies | `results/recovery.json` |
| `graph_bench.py` | Reference `Store` scaling: ingest, temporal query, conflicts, joins, patches, save/load | `results/graph_bench.json` |
| `representation_compare.py` | Legacy TCIR (sandboxed) vs proposed records on the same object graph | `results/representation.json`, `.txt` |

Narrative results and limitations are in [`docs/revival/06-evaluation.md`](../../docs/revival/06-evaluation.md).

## Data (not committed)

| File | Source | License | sha256 |
| --- | --- | --- | --- |
| `banking77_train.csv` | `https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv` | CC-BY-4.0 | `b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b` |
| `banking77_test.csv` | same repository, `test.csv` | CC-BY-4.0 | `d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d` |
| `hotpot_distractor_validation.parquet` | `https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/distractor/validation-00000-of-00001.parquet` | CC-BY-SA-4.0 | `c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6` |

## Protocol notes

- **Held-out data.**
  - **Banking77:** the test split was used only by `banking77_cascade.py`. The keyword rules were written from training examples and revised once (v1 → v2) after reviewing errors on the 10% training holdout. After that revision the holdout is no longer clean for the rules, so rules precision is reported on test only.
  - **Learned tier:** its threshold and temperature were fit on that holdout before the test split was read.
- **Energy.** The model-tier energy figures come from `nvidia-smi --query-gpu=power.draw`, sampled every 250 ms, minus a 3 s idle baseline.
- **Legacy probe.** `legacy_probe/make_sandbox.py` lists each import-only shim needed to run the legacy TCIR. The legacy half should run with pydantic 2.5.0 (the version the legacy `pyproject.toml` pins).
