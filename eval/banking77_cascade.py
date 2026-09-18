"""Evaluate the support router's `classify` step on Banking77 under three bindings.

    python eval/banking77_cascade.py --data-dir DIR [--model Qwen/Qwen3-8B] [--llm-subset 500]

Measures, on the official 3,080-example test split (never used for fitting,
threshold selection, or rule writing):
  * quality: coverage, selective accuracy (Wilson 95% CI), calibration (ECE)
  * latency: per-call distributions, split into backend time and TensaCode overhead
  * throughput: batched vs per-item
  * memory: process RSS and CUDA peak allocation
  * cost: metered USD (none: all local) and GPU-reported energy for the model tier
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import tensacode as tc  # noqa: E402
from examples.support_router import config  # noqa: E402
from examples.support_router.domain import Intent  # noqa: E402
from tensacode.runtime import FunctionImplementation, Output  # noqa: E402


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def pct(xs: list[float]) -> dict[str, float]:
    a = np.asarray(xs, dtype=float)
    return {"n": len(a), "p50": round(float(np.percentile(a, 50)), 4), "p95": round(float(np.percentile(a, 95)), 4), "p99": round(float(np.percentile(a, 99)), 4), "mean": round(float(a.mean()), 4)}


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return round(float(total), 4)


def selective(preds: list, gold: list) -> dict:
    answered = [(p, g) for p, g in zip(preds, gold) if not isinstance(p, tc.Unknown)]
    k = sum(p == g for p, g in answered)
    return {
        "n": len(gold),
        "answered": len(answered),
        "coverage": round(len(answered) / len(gold), 4),
        "selective_accuracy": round(k / len(answered), 4) if answered else None,
        "selective_accuracy_ci95": wilson(k, len(answered)),
        "errors": len(answered) - k,
    }


class PowerSampler:
    """GPU board power as reported by nvidia-smi (GB10: SoC GPU rail). Excludes CPU, memory, and system."""

    def __init__(self, period_ms: int = 250) -> None:
        self.samples: list[tuple[float, float]] = []
        self._proc = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits", f"-lms={period_ms}"],
            stdout=subprocess.PIPE,
            text=True,
        )
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()

    def _read(self) -> None:
        for line in self._proc.stdout:  # type: ignore[union-attr]
            try:
                self.samples.append((time.monotonic(), float(line.strip())))
            except ValueError:
                pass

    def joules(self, t0: float, t1: float, baseline_w: float = 0.0) -> float | None:
        pts = [(t, w) for t, w in self.samples if t0 <= t <= t1]
        if len(pts) < 2:
            return None
        return float(sum((b[0] - a[0]) * ((a[1] + b[1]) / 2 - baseline_w) for a, b in zip(pts, pts[1:])))

    def mean_watts(self, t0: float, t1: float) -> float | None:
        pts = [w for t, w in self.samples if t0 <= t <= t1]
        return float(np.mean(pts)) if pts else None

    def stop(self) -> None:
        self._proc.terminate()


def environment(data_dir: Path) -> dict:
    import sklearn
    import torch
    import transformers

    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip())
    cpu = next((l.split(":", 1)[1].strip() for l in subprocess.run(["lscpu"], capture_output=True, text=True).stdout.splitlines() if l.startswith("Model name")), platform.processor())
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": cpu,
        "cpu_count": os.cpu_count(),
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "scikit_learn": sklearn.__version__,
        "numpy": np.__version__,
        "tensacode_py_commit": git + ("+uncommitted-proposal" if dirty else ""),
        "dataset": {
            f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(data_dir.glob("banking77_*.csv"))
        },
        "dataset_source": "https://github.com/PolyAI-LDN/task-specific-datasets (CC-BY-4.0)",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--llm-subset", type=int, default=500)
    ap.add_argument("--out", type=Path, default=ROOT / "eval/results/banking77.json")
    args = ap.parse_args()
    random.seed(0)
    results: dict = {"environment": environment(args.data_dir)}
    proc = psutil.Process()

    test = config.load_banking77(args.data_dir / "banking77_test.csv")
    texts, gold = [t for t, _ in test], [y for _, y in test]
    print(f"test examples: {len(test)}")

    # ---- fit learned tier (train split minus 10% holdout; test untouched)
    learned, fit = config.learned_classifier(args.data_dir / "banking77_train.csv")
    results["learned_fit"] = asdict(fit)
    results["memory_after_fit_rss_mb"] = round(proc.memory_info().rss / 2**20, 1)
    print("fit:", fit)

    # ---- tiers in isolation (direct calls, no runtime)
    reqs = [tc.Request("classify", t, Intent) for t in texts]
    t0 = time.perf_counter()
    rule_out = [o.value for o in config.KEYWORD_RULES.run(reqs)]
    rules_s = time.perf_counter() - t0
    results["rules_alone"] = selective(rule_out, gold) | {"direct_batch_seconds": round(rules_s, 4)}

    probs = learned.probabilities(texts)
    classes = learned.model.classes_
    pred = [Intent(classes[i]) for i in probs.argmax(1)]
    conf = probs.max(1)
    correct = np.array([p == g for p, g in zip(pred, gold)])
    raw_logits = learned.model.decision_function(learned.features.transform(texts))
    raw_probs = np.exp(raw_logits - raw_logits.max(1, keepdims=True))
    raw_probs /= raw_probs.sum(1, keepdims=True)
    learned_sel = [p if c >= learned.threshold else tc.Unknown("below_threshold") for p, c in zip(pred, conf)]
    results["learned_alone"] = {
        "accuracy_no_abstention": round(float(correct.mean()), 4),
        "accuracy_ci95": wilson(int(correct.sum()), len(correct)),
        "ece_temperature_scaled": ece(conf, correct),
        "ece_unscaled": ece(raw_probs.max(1), np.array([Intent(classes[i]) == g for i, g in zip(raw_probs.argmax(1), gold)])),
        "at_validation_threshold": selective(learned_sel, gold) | {"threshold": round(learned.threshold, 4), "target": fit.target_accuracy},
    }

    # ---- TensaCode runtime: rules -> learned, per item and batched
    policy = tc.Policy(localities=frozenset({"in_process"}), available=frozenset({"sklearn"}), cache=False)
    rt = tc.Runtime([config.KEYWORD_RULES, learned], policy=policy)
    with tc.use(rt):
        for t in texts[:50]:  # warm-up, excluded
            tc.classify(t, Intent)
        rt.trace.spans.clear()
        t0 = time.perf_counter()
        per_item = [tc.classify(t, Intent) for t in texts]
        per_item_s = time.perf_counter() - t0
    spans = rt.trace.spans
    tier = [s.answered_by or "unknown" for s in spans]
    results["cascade_rules_learned"] = selective(per_item, gold) | {
        "answered_by": {k: tier.count(k) for k in sorted(set(tier))},
        "per_item_latency_ms": {
            "total": pct([s.total_ms for s in spans]),
            "backend": pct([s.backend_ms for s in spans]),
            "tensacode_overhead": pct([s.total_ms - s.backend_ms for s in spans]),
        },
        "per_item_throughput_per_s": round(len(texts) / per_item_s, 1),
    }
    rt_batch = tc.Runtime([config.KEYWORD_RULES, learned], policy=policy)
    with tc.use(rt_batch):
        t0 = time.perf_counter()
        batched = []
        for i in range(0, len(texts), 256):
            batched += tc.classify.many(texts[i : i + 256], Intent)
        batch_s = time.perf_counter() - t0
    assert [type(x) for x in batched] == [type(x) for x in per_item]
    results["cascade_rules_learned"]["batched_256_throughput_per_s"] = round(len(texts) / batch_s, 1)
    results["cascade_rules_learned"]["batched_equals_per_item"] = batched == per_item

    # direct per-item baseline (same work, no runtime) for overhead cross-check
    t0 = time.perf_counter()
    for r in reqs:
        out = config.KEYWORD_RULES.run([r])[0]
        if isinstance(out.value, tc.Unknown):
            learned.run([r])
    results["direct_rules_then_learned_per_item_throughput_per_s"] = round(len(texts) / (time.perf_counter() - t0), 1)

    # ---- pure runtime overhead with trivial implementations
    noop_answer = FunctionImplementation("classify", "noop-answer", "1", lambda r: Output(Intent.card_arrival))
    noop_abstain = FunctionImplementation("classify", "noop-abstain", "1", lambda r: Output(tc.Unknown("x")))
    overhead = {}
    for label, impls in {"1_tier_answer": [noop_answer], "3_tiers_2_abstain": [noop_abstain, FunctionImplementation("classify", "noop-abstain-2", "1", lambda r: Output(tc.Unknown("y"))), noop_answer]}.items():
        rt_o = tc.Runtime(impls, policy=tc.Policy(cache=False))
        with tc.use(rt_o):
            for t in texts[:2000]:
                tc.classify(t, Intent)
        ms = [s.total_ms for s in rt_o.trace.spans]
        overhead[label] = pct(ms)
    results["runtime_overhead_trivial_impls_ms"] = overhead
    results["memory_after_runtime_rss_mb"] = round(proc.memory_info().rss / 2**20, 1)

    # ---- optional: local general model
    if args.model:
        import torch

        sampler = PowerSampler()
        time.sleep(3)
        idle_w = sampler.mean_watts(time.monotonic() - 3, time.monotonic())
        t0 = time.perf_counter()
        model = config.local_model_classifier(args.model)
        load_s = time.perf_counter() - t0
        results["model"] = {"id": args.model, "prompt_version": model.version, "load_seconds": round(load_s, 1), "gpu_idle_watts": idle_w}

        subset = random.Random(0).sample(range(len(texts)), args.llm_subset)
        sub_reqs = [reqs[i] for i in subset]
        torch.cuda.reset_peak_memory_stats()
        m0, w0 = time.monotonic(), time.perf_counter()
        outs = [o.value for o in model.run(sub_reqs)]
        m1, llm_s = time.monotonic(), time.perf_counter() - w0
        sub_gold = [gold[i] for i in subset]
        declined = sum(isinstance(o, tc.Unknown) and o.reason == "model_declined" for o in outs)
        unparseable = sum(isinstance(o, tc.Unknown) and o.reason == "unparseable_output" for o in outs)
        results["model_alone_subset"] = selective(outs, sub_gold) | {
            "subset_seed": 0,
            "declined": declined,
            "unparseable": unparseable,
            "batch_size": model.batch_size,
            "seconds": round(llm_s, 2),
            "amortized_ms_per_item": round(1e3 * llm_s / len(sub_reqs), 1),
            "gpu_energy_j_per_item_above_idle": (lambda j: round(j / len(sub_reqs), 2) if j is not None else None)(sampler.joules(m0, m1, idle_w or 0.0)),
            "gpu_mean_watts": sampler.mean_watts(m0, m1),
            "cuda_peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1),
            "learned_tier_accuracy_on_same_subset": round(float(np.mean([pred[i] == gold[i] for i in subset])), 4),
        }
        single = []
        for r in sub_reqs[:40]:
            s0 = time.perf_counter()
            model.run([r])
            single.append(1e3 * (time.perf_counter() - s0))
        results["model_alone_subset"]["batch1_latency_ms"] = pct(single[5:])

        policy_m = tc.Policy(localities=frozenset({"in_process"}), available=frozenset({"sklearn", "cuda"}), cache=False)
        rt_m = tc.Runtime([config.KEYWORD_RULES, learned, model], policy=policy_m)
        m0 = time.monotonic()
        with tc.use(rt_m):
            full = []
            t0 = time.perf_counter()
            for i in range(0, len(texts), 128):
                full += tc.classify.many(texts[i : i + 128], Intent)
            full_s = time.perf_counter() - t0
        m1 = time.monotonic()
        spans_m = rt_m.trace.spans
        tier_m = [s.answered_by or "unknown" for s in spans_m]
        escalated = [i for i, s in enumerate(spans_m) if any(a.implementation.startswith("chat:") for a in s.attempts)]
        esc_ans = [(full[i], gold[i]) for i in escalated if not isinstance(full[i], tc.Unknown)]
        results["cascade_rules_learned_model"] = selective(full, gold) | {
            "answered_by": {k: tier_m.count(k) for k in sorted(set(tier_m))},
            "escalated_to_model": len(escalated),
            "model_answered_escalations": len(esc_ans),
            "model_accuracy_on_escalations": round(sum(p == g for p, g in esc_ans) / len(esc_ans), 4) if esc_ans else None,
            "model_accuracy_on_escalations_ci95": wilson(sum(p == g for p, g in esc_ans), len(esc_ans)),
            "learned_argmax_accuracy_on_same_escalations": round(float(np.mean([pred[i] == gold[i] for i in escalated])), 4) if escalated else None,
            "seconds_total_batched_128": round(full_s, 2),
            "gpu_energy_j_above_idle_total": sampler.joules(m0, m1, idle_w or 0.0),
        }
        sampler.stop()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, default=str))
    print(json.dumps({k: v for k, v in results.items() if k != "environment"}, indent=2, default=str))


if __name__ == "__main__":
    main()
