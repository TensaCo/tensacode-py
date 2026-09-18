"""Learned label encoder against the hand-written rules, on data neither of them was fitted to.

    PYTHONPATH=src:. python eval/training/eval_label_encoder.py --model DIR --data DIR --cache DIR

Three measurements, each with a baseline that is the code currently in the live path:

1. **Form concepts, held-out phrasing.** The access agent scores a field label against
   hand-written synonym lists with ``label_similarity``. The app renders four label
   variants per field and the synonym lists contain three, so the fourth is a
   generalisation test nobody wrote for this purpose. Reported: top-1 over the four
   concepts, the score of the right concept, and whether it clears the 0.3 threshold the
   rule needs before it will even claim ``may_mean``.
2. **Targeting through a recognizer.** For every captured frame, a control is named by the
   DOM (truth) and read by the recognizer (noisy). Query with the DOM name, rank the
   recognizer's readings, and count a hit when the top-ranked reading sits inside the
   control's own box. Splits: ``tune`` (the encoder's OCR pairs came from here),
   ``test_app`` (apps it never saw), ``test_os`` (macOS and Windows).
3. **Refusal behaviour.** ``find`` abstains when the best score is under 0.55 or the top
   two are within 0.08. Both scorers are run through that same rule, so the comparison is
   of scorers, not of thresholds: hit, miss, and refused are reported separately.
"""

from __future__ import annotations

import argparse
import json
import pickle
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).parents[2] / "src"), str(Path(__file__).parents[2])]

from examples.browser_agents.browser import label_similarity  # noqa: E402
from examples.browser_agents.learned.text_embed import LabelEncoder, normalize  # noqa: E402
from examples.browser_agents.tasks.access_script import TEXT_CONCEPTS, _type_evidence  # noqa: E402
from eval.training.train_label_encoder import APP_LABELS, HELD_OUT  # noqa: E402

MIN_SCORE, MARGIN = 0.55, 0.08  # browser.find's own thresholds


def concept_scores_rules(label: str) -> dict[str, float]:
    return {c: max(label_similarity(s, label) for s in syn) for c, syn in TEXT_CONCEPTS.items()}


def concept_scores_learned(encoder: LabelEncoder, label: str) -> dict[str, float]:
    known = {c: [s for s in syn] + APP_LABELS[c][:-1] for c, syn in TEXT_CONCEPTS.items()}
    v = encoder.encode([label])[0]
    out = {}
    for concept, phrasings in known.items():
        m = encoder.encode(phrasings)
        out[concept] = float((m @ v).max())
    return out


def concepts_table(encoder: LabelEncoder) -> dict:
    rows = []
    for concept, variants in APP_LABELS.items():
        for label in variants:
            held = label == HELD_OUT[concept]
            for arm, scores in (("rules", concept_scores_rules(label)), ("learned", concept_scores_learned(encoder, label))):
                order = sorted(scores.items(), key=lambda kv: -kv[1])
                rows.append({
                    "label": label, "truth": concept, "held_out": held, "arm": arm,
                    "picked": order[0][0], "correct": order[0][0] == concept,
                    "score_of_truth": round(scores[concept], 4),
                    "margin": round(order[0][1] - order[1][1], 4),
                    "clears_claim_threshold": scores[concept] >= 0.3,
                })
    summary = {}
    for arm in ("rules", "learned"):
        for split, want in (("known", False), ("held_out", True)):
            sel = [r for r in rows if r["arm"] == arm and r["held_out"] == want]
            summary[f"{arm}/{split}"] = {
                "n": len(sel),
                "top1": round(sum(r["correct"] for r in sel) / len(sel), 4),
                "mean_score_of_truth": round(statistics.mean(r["score_of_truth"] for r in sel), 4),
                "mean_margin": round(statistics.mean(r["margin"] for r in sel), 4),
                "clears_claim_threshold": round(sum(r["clears_claim_threshold"] for r in sel) / len(sel), 4),
            }
    return {"rows": rows, "summary": summary}


def _inside(box: tuple[int, int, int, int], target: tuple[int, int, int, int]) -> bool:
    x, y, w, h = box
    tx, ty, tw, th = target
    cx, cy = x + w / 2, y + h / 2
    return tx <= cx <= tx + tw and ty <= cy <= ty + th


def targeting(encoder: LabelEncoder, data: Path, cache: Path) -> dict:
    """Rank recognizer readings for a DOM-named control, and see where the top one sits."""
    per_split: dict[str, dict[str, int]] = {}
    decisions: dict[str, list[tuple[float, int]]] = {}
    latency = {"rules": [], "learned": []}
    for meta_path in sorted(data.glob("*.json")):
        frame = json.loads(meta_path.read_text())
        split = frame["meta"].get("split") or "unknown"
        blob = cache / f"{meta_path.stem}.pkl"
        if not blob.exists():
            continue
        words = pickle.loads(blob.read_bytes()).get("words") or []
        # group words into lines, which is what a pixel provider offers as an element name
        lines: dict[int, list] = {}
        for w in words:
            if getattr(w, "text", "").strip():
                lines.setdefault(getattr(w, "line", 0), []).append(w)
        candidates = []
        for group in lines.values():
            xs = [w.box[0] for w in group]
            ys = [w.box[1] for w in group]
            x2 = max(w.box[0] + w.box[2] for w in group)
            y2 = max(w.box[1] + w.box[3] for w in group)
            candidates.append((" ".join(w.text for w in group), (min(xs), min(ys), x2 - min(xs), y2 - min(ys))))
        if not candidates:
            continue
        names = [c[0] for c in candidates]
        emb = encoder.encode(names)
        for control in frame["screen"]["controls"]:
            truth = (control.get("name") or "").strip()
            if not truth or len(truth) > 40 or control.get("disabled"):
                continue
            if not any(_inside(box, tuple(control["box"])) for _, box in candidates):
                continue  # the recognizer read nothing inside this control: not a targeting question
            for arm in ("rules", "learned"):
                t0 = time.perf_counter()
                if arm == "rules":
                    scores = np.array([label_similarity(truth, n) for n in names])
                else:
                    scores = emb @ encoder.encode([truth])[0]
                latency[arm].append((time.perf_counter() - t0) * 1000)
                order = np.argsort(-scores)
                best, second = float(scores[order[0]]), float(scores[order[1]]) if len(order) > 1 else -1.0
                bucket = per_split.setdefault(f"{arm}/{split}", {"hit": 0, "miss": 0, "refused": 0})
                decisions.setdefault(f"{arm}/{split}", []).append((best, int(_inside(candidates[order[0]][1], tuple(control["box"])))))
                if best < MIN_SCORE or (second >= 0 and best - second < MARGIN and names[order[0]] != names[order[1]]):
                    bucket["refused"] += 1
                elif _inside(candidates[order[0]][1], tuple(control["box"])):
                    bucket["hit"] += 1
                else:
                    bucket["miss"] += 1
    out = {}
    for key, b in sorted(per_split.items()):
        total = sum(b.values())
        out[key] = {**b, "n": total, "hit_rate": round(b["hit"] / total, 4), "wrong_rate": round(b["miss"] / total, 4), "refused_rate": round(b["refused"] / total, 4)}
    curves = {}
    for split in sorted({k.split("/")[1] for k in decisions}):
        curves[split] = equal_coverage(decisions, split)
    return {"targeting": out, "equal_coverage": curves,
            "latency_ms": {k: {"p50": round(statistics.median(v), 4), "p95": round(sorted(v)[int(0.95 * len(v))], 4), "n": len(v)} for k, v in latency.items() if v}}


def equal_coverage(decisions: dict, split: str) -> dict:
    """Each arm's own thresholds are meaningless across scorers; coverage is comparable.

    For every arm, sweep the accept threshold over its own score distribution and report
    accuracy among accepted items at matched coverage. A scorer is better only if it is
    more accurate at the same coverage.
    """
    out: dict[str, dict] = {}
    targets = (0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0)
    for arm in ("rules", "learned"):
        rows = decisions.get(f"{arm}/{split}") or []
        if not rows:
            continue
        rows = sorted(rows, key=lambda r: -r[0])  # (best score, correct)
        n = len(rows)
        at = {}
        for target in targets:
            k = max(1, int(round(target * n)))
            acc = sum(c for _, c in rows[:k]) / k
            at[f"cov={target:.1f}"] = round(acc, 4)
        out[arm] = {"n": n, "accuracy_at_coverage": at, "accuracy_at_full_coverage": round(sum(c for _, c in rows) / n, 4)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("eval/results/learned_label_matching.json"))
    args = ap.parse_args()

    encoder = LabelEncoder.load(args.model / "label_encoder.npz")
    manifest = json.loads((args.model / "label_encoder.manifest.json").read_text())
    report = {"manifest": manifest, "concepts": concepts_table(encoder), **targeting(encoder, args.data, args.cache)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))

    print("== form concepts (top-1 / mean score of the right concept / clears the 0.3 claim threshold)")
    for key, s in report["concepts"]["summary"].items():
        print(f"  {key:<18} n={s['n']:<3} top1={s['top1']:.3f}  score={s['mean_score_of_truth']:.3f}  margin={s['mean_margin']:.3f}  clears={s['clears_claim_threshold']:.3f}")
    print("== held-out phrasings, one line each")
    for r in report["concepts"]["rows"]:
        if r["held_out"]:
            print(f"  {r['arm']:<8} {r['label']:<20} -> {r['picked']:<12} {'OK' if r['correct'] else 'WRONG':<6} score={r['score_of_truth']:.3f} margin={r['margin']:.3f}")
    print("== targeting through the recognizer")
    for key, s in report["targeting"].items():
        print(f"  {key:<18} n={s['n']:<5} hit={s['hit_rate']:.3f}  wrong={s['wrong_rate']:.3f}  refused={s['refused_rate']:.3f}")
    print("== accuracy at matched coverage (each arm thresholded on its own scores)")
    for split, arms in report["equal_coverage"].items():
        for arm, s in arms.items():
            row = "  ".join(f"{k}:{v:.3f}" for k, v in s["accuracy_at_coverage"].items())
            print(f"  {arm:<8} {split:<9} n={s['n']:<5} {row}")
    print("== latency (ms per ranking call)")
    for key, s in report["latency_ms"].items():
        print(f"  {key:<8} p50={s['p50']:.3f} p95={s['p95']:.3f} n={s['n']}")


if __name__ == "__main__":
    main()
