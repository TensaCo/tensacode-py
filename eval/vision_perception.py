"""Evaluate pixels -> scene graph against DOM ground truth captured at the same instant.

    PYTHONPATH=src:. python eval/vision_perception.py --data DIR [--cache DIR] [--out eval/results/vision_perception.json]

Frames come from ``eval/vision_capture.py``. Ground truth per frame:
* words: DOM text split into words, kept only if hit-testable at their own location
  (text under another window does not count);
* controls: DOM controls with an unobstructed click point, excluding invisible window
  resize handles;
* windows: Seed ``app-window`` frames with at least 30% of their area not covered.

Metrics (per split: tune / test_app / test_os):
* word recall and precision: same normalized text and IoU >= 0.5 (``exact``), or text
  similarity >= 0.8 and IoU >= 0.5 (``fuzzy``);
* control detection: a prediction matches a DOM control if its click point is inside the
  control box or IoU >= 0.5 (one-to-one); recall by coarse role, precision overall and
  by rule, and role accuracy on matches;
* targeting: for each DOM control with a unique, short name, look it up among the
  predicted controls the way agents do (``browser.label_similarity``, min 0.55, margin
  0.08) and check the predicted click point falls inside the DOM box. Split into
  controls whose name is visible as text inside them (``text``), icon-only buttons
  (``icon``), and fields whose label is outside the box (``field``);
* windows: IoU >= 0.5 with a visible DOM window, or inside it covering >= 35% (its visible
  part), and title equality ignoring case and punctuation.
Model outputs (OCR words, detector boxes) are cached per frame, so rule changes can be
re-evaluated without re-running models; latencies come from the uncached run.
"""

from __future__ import annotations

import argparse
import difflib
import json
import pickle
import platform
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).parents[1] / "src"), str(Path(__file__).parents[1])]

from examples.browser_agents.browser import label_similarity  # noqa: E402
from examples.browser_agents.vision.icon_memory import IconMemory  # noqa: E402
from examples.browser_agents.vision.perceive import center, contains, inside, iou, perceive  # noqa: E402

OUT = Path(__file__).parent / "results" / "vision_perception.json"
TEXTBOX = {"textbox", "searchbox", "input"}
CHECK = {"checkbox", "radio", "switch"}


def norm(s: str) -> str:
    return re.sub(r"[^\w@.:/~$#-]+", "", s.casefold()).strip(".:,;")


def coarse(role: str) -> str:
    return "textbox" if role in TEXTBOX else "checkbox" if role in CHECK else "button"


def load_frames(data: Path) -> list[dict]:
    frames = []
    for j in sorted(data.glob("f*.json")):
        d = json.loads(j.read_text())
        d["name"], d["png"] = j.stem, j.with_suffix(".png")
        frames.append(d)
    return frames


def gt_words(frame: dict) -> list[dict]:
    return [w for w in frame["words"] if re.search(r"[A-Za-z0-9]", w["text"])]


def gt_controls(frame: dict) -> list[dict]:
    out = []
    for c in frame["screen"]["controls"]:
        x, y, w, h = c["box"]
        if not c["point"] or w * h < 16 or c["name"].startswith("Resize window"):
            continue
        out.append(c)
    return out


def visible_windows(frame: dict) -> list[dict]:
    wins = sorted(frame["windows"], key=lambda w: w["z"])
    out = []
    for i, w in enumerate(wins):
        x, y, ww, hh = w["box"]
        mask = np.ones((max(1, hh), max(1, ww)), bool)
        for above in wins[i + 1:]:
            ax, ay, aw, ah = above["box"]
            x0, y0 = max(0, ax - x), max(0, ay - y)
            x1, y1 = min(ww, ax + aw - x), min(hh, ay + ah - y)
            if x1 > x0 and y1 > y0:
                mask[y0:y1, x0:x1] = False
        if mask.mean() >= 0.3:
            out.append(w)
    return out


def name_visible(c: dict, words: list[dict]) -> bool:
    inside_words = {norm(w["text"]) for w in words if inside(center(tuple(w["box"])), tuple(c["box"]), pad=2)}
    tokens = [norm(t) for t in c["name"].split()[:6] if norm(t)]
    return bool(tokens) and sum(t in inside_words for t in tokens) >= 0.5 * len(tokens)


# ------------------------------------------------------------------ models (cached)


def model_outputs(frames: list[dict], cache: Path) -> tuple[dict, dict]:
    cache.mkdir(parents=True, exist_ok=True)
    todo = [f for f in frames if not (cache / f"{f['name']}.pkl").exists()]
    load = {}
    if todo:
        from PIL import Image

        from examples.browser_agents.vision.models import DoctrOCR, IconDetector

        ocr, det = DoctrOCR(), IconDetector()
        ocr.load()
        det.load()
        load = {"ocr": vars(ocr.stats), "detector": vars(det.stats)}
        warm = np.asarray(Image.open(todo[0]["png"]).convert("RGB"))
        ocr(warm), det(warm)
        for f in todo:
            rgb = np.asarray(Image.open(f["png"]).convert("RGB"))
            t0 = time.perf_counter()
            words = ocr(rgb)
            t1 = time.perf_counter()
            icons = det(rgb)
            t2 = time.perf_counter()
            (cache / f"{f['name']}.pkl").write_bytes(pickle.dumps({"words": words, "icons": icons, "ocr_ms": (t1 - t0) * 1e3, "det_ms": (t2 - t1) * 1e3}))
        (cache / "load.json").write_text(json.dumps(load))
        ocr.free()
        det.free()
    load = json.loads((cache / "load.json").read_text()) if (cache / "load.json").exists() else load
    return {f["name"]: pickle.loads((cache / f"{f['name']}.pkl").read_bytes()) for f in frames}, load


# ------------------------------------------------------------------ metrics


def match_words(pred: list, gold: list[dict]) -> tuple[int, int]:
    """(exact matches, fuzzy matches), one-to-one."""
    exact = fuzzy = 0
    used: set[int] = set()
    for g in gold:
        gb, gt = tuple(g["box"]), norm(g["text"])
        best, kind = None, 0
        for i, p in enumerate(pred):
            if i in used or iou(p.box, gb) < 0.5:
                continue
            pt = norm(p.text)
            if pt == gt:
                best, kind = i, 2
                break
            if difflib.SequenceMatcher(None, pt, gt).ratio() >= 0.8 and kind < 1:
                best, kind = i, 1
        if best is not None:
            used.add(best)
            exact += kind == 2
            fuzzy += 1
    return exact, fuzzy


def match_controls(pred: list, gold: list[dict]) -> list[tuple[int, int]]:
    pairs = []
    for gi, g in enumerate(gold):
        gb = tuple(g["box"])
        for pi, p in enumerate(pred):
            c = p.control
            score = iou(c.box, gb) + (1.0 if inside(c.point, gb) else 0.0)
            if score >= 0.5:
                pairs.append((score, gi, pi))
    pairs.sort(reverse=True)
    used_g, used_p, out = set(), set(), []
    for _, gi, pi in pairs:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        out.append((gi, pi))
    return out


def target(pred: list, query: str) -> object | None:
    scored = sorted(((label_similarity(query, p.control.name), p) for p in pred if p.control.name), key=lambda sp: -sp[0])
    if not scored or scored[0][0] < 0.55:
        return None
    if len(scored) > 1 and scored[0][0] - scored[1][0] < 0.08 and scored[0][1].control.name != scored[1][1].control.name:
        return None
    return scored[0][1]


def evaluate(frames: list[dict], outputs: dict, *, icon_memory: IconMemory | None, text_candidates: bool, overlays: Path | None = None) -> dict:
    from PIL import Image

    agg: dict = defaultdict(lambda: defaultdict(float))
    per_rule_pred, per_rule_hit = Counter(), Counter()
    latency = defaultdict(list)
    for f in frames:
        rgb = np.asarray(Image.open(f["png"]).convert("RGB"))
        cached = outputs[f["name"]]
        t0 = time.perf_counter()
        scene = perceive(rgb, ocr=lambda _: cached["words"], detector=lambda _: cached["icons"], icon_namer=icon_memory, text_candidates=text_candidates)
        rules_ms = (time.perf_counter() - t0) * 1e3
        latency["ocr_ms"].append(cached["ocr_ms"])
        latency["detector_ms"].append(cached["det_ms"])
        latency["cv_and_rules_ms"].append(rules_ms)
        latency["total_ms"].append(cached["ocr_ms"] + cached["det_ms"] + rules_ms)
        split = f["meta"]["split"]
        a = agg[split]
        a["frames"] += 1
        # words
        gw = gt_words(f)
        pw = [w for w in cached["words"] if re.search(r"[A-Za-z0-9]", w.text)]
        exact, fuzzy = match_words(pw, gw)
        a["gt_words"] += len(gw)
        a["pred_words"] += len(pw)
        a["word_exact"] += exact
        a["word_fuzzy"] += fuzzy
        # controls
        gc = gt_controls(f)
        pairs = match_controls(scene.controls, gc)
        matched_p = {pi for _, pi in pairs}
        a["gt_controls"] += len(gc)
        a["pred_controls"] += len(scene.controls)
        a["matched_controls"] += len(pairs)
        for gi, pi in pairs:
            role = coarse(gc[gi]["role"])
            a[f"matched_{role}"] += 1
            a[f"role_ok_{role}"] += scene.controls[pi].control.role == role
        for c in gc:
            a[f"gt_{coarse(c['role'])}"] += 1
        for pi, p in enumerate(scene.controls):
            per_rule_pred[(split, p.rule)] += 1
            per_rule_hit[(split, p.rule)] += pi in matched_p
        # targeting by name
        names = Counter(c["name"] for c in gc)
        for c in gc:
            if not c["name"] or names[c["name"]] > 1 or len(c["name"]) > 40:
                continue
            kind = "text" if name_visible(c, gw) else "icon" if coarse(c["role"]) == "button" and c["role"] != "combobox" else "field"
            hit = target(scene.controls, c["name"])
            a[f"target_{kind}_n"] += 1
            a[f"target_{kind}_ok"] += hit is not None and inside(hit.control.point, tuple(c["box"]))
            a[f"target_{kind}_wrong"] += hit is not None and not inside(hit.control.point, tuple(c["box"]))
        # windows
        gwin = visible_windows(f)
        a["gt_windows"] += len(gwin)
        a["pred_windows"] += len(scene.windows)
        for w in gwin:
            wb = tuple(w["box"])
            # a partly covered window can only be seen as its visible part: inside the frame and covering enough of it
            m = [p for p in scene.windows if iou(p.box, wb) >= 0.5 or (contains(wb, p.box, pad=10) and p.box[2] * p.box[3] >= 0.35 * wb[2] * wb[3])]
            a["window_found"] += bool(m)
            a["window_title_ok"] += any(re.sub(r"\W+", "", p.title.casefold()) == re.sub(r"\W+", "", w["title"].casefold()) for p in m)
        if overlays is not None:
            draw_overlay(rgb, scene, overlays / f"{f['name']}_{split}.png")
    report = {}
    for split, a in agg.items():
        r = lambda n, d: round(a[n] / a[d], 3) if a[d] else None  # noqa: E731
        report[split] = {
            "frames": int(a["frames"]),
            "words": {"gt": int(a["gt_words"]), "recall_exact": r("word_exact", "gt_words"), "recall_fuzzy": r("word_fuzzy", "gt_words"), "precision_fuzzy": r("word_fuzzy", "pred_words")},
            "controls": {
                "gt": int(a["gt_controls"]), "predicted": int(a["pred_controls"]),
                "recall": r("matched_controls", "gt_controls"), "precision": r("matched_controls", "pred_controls"),
                "recall_by_role": {k: {"gt": int(a[f"gt_{k}"]), "recall": r(f"matched_{k}", f"gt_{k}"), "role_accuracy_on_matches": r(f"role_ok_{k}", f"matched_{k}")} for k in ("button", "textbox", "checkbox")},
            },
            "targeting_by_name": {
                kind: {"queries": int(a[f"target_{kind}_n"]), "clicks_right_control": r(f"target_{kind}_ok", f"target_{kind}_n"), "clicks_wrong_control": r(f"target_{kind}_wrong", f"target_{kind}_n")}
                for kind in ("text", "icon", "field")
            },
            "windows": {"gt_visible": int(a["gt_windows"]), "predicted": int(a["pred_windows"]), "found": r("window_found", "gt_windows"), "title_correct": r("window_title_ok", "gt_windows")},
            "precision_by_rule": {rule: {"predicted": n, "matched_a_dom_control": round(per_rule_hit[(s, rule)] / n, 3)} for (s, rule), n in sorted(per_rule_pred.items()) if s == split},
        }
    report["latency_ms_per_frame"] = {k: {"p50": round(statistics.median(v), 1), "p95": round(sorted(v)[int(0.95 * (len(v) - 1))], 1)} for k, v in latency.items()}
    return report


def draw_overlay(rgb: np.ndarray, scene, path: Path) -> None:
    import cv2
    from PIL import Image

    img = rgb.copy()
    colors = {"button": (230, 120, 0), "textbox": (0, 150, 255), "checkbox": (210, 0, 210)}
    for w in scene.windows:
        x, y, ww, hh = w.box
        cv2.rectangle(img, (x, y), (x + ww, y + hh), (0, 200, 0), 3)
        cv2.putText(img, w.title[:30], (x + 4, y + hh - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 0), 1)
    for inf in scene.controls:
        c = inf.control
        x, y, ww, hh = c.box
        col = (250, 210, 0) if inf.rule == "short-text-candidate" else colors.get(c.role, (255, 0, 0))
        cv2.rectangle(img, (x, y), (x + ww, y + hh), col, 1 if inf.rule == "short-text-candidate" else 2)
        cv2.circle(img, c.point, 2, (255, 0, 0), -1)
        if inf.rule == "icon-memory":
            cv2.putText(img, c.name[:14], (x, y + hh + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 60, 60), 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path)


def fit_icon_memory(frames: list[dict]) -> IconMemory:
    from PIL import Image

    memory = IconMemory()
    for f in frames:
        rgb = np.asarray(Image.open(f["png"]).convert("RGB"))
        gw = gt_words(f)
        for c in gt_controls(f):
            x, y, w, h = c["box"]
            if c["name"] and 12 <= w <= 72 and 12 <= h <= 72 and not name_visible(c, gw):
                memory.remember(rgb, tuple(c["box"]), c["name"])
    return memory


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--overlays", type=Path, default=None)
    ap.add_argument("--splits", default="tune,test_app,test_os")
    ap.add_argument("--icon-memory", type=Path, default=None, help="use this icon memory instead of fitting one on the tune split (e.g. one learned from tooltips)")
    args = ap.parse_args()
    frames = [f for f in load_frames(args.data) if f["meta"]["split"] in args.splits.split(",")]
    outputs, load = model_outputs(frames, args.cache or args.data / "_model_cache")
    memory = IconMemory.load(args.icon_memory) if args.icon_memory else fit_icon_memory([f for f in frames if f["meta"]["split"] == "tune"])
    variants = {
        "full (text candidates + icon memory fit on tune)": dict(icon_memory=memory, text_candidates=True),
        "no icon memory": dict(icon_memory=None, text_candidates=True),
        "no text candidates": dict(icon_memory=memory, text_candidates=False),
    }
    results = {}
    for label, kw in variants.items():
        results[label] = evaluate(frames, outputs, overlays=args.overlays if label.startswith("full") else None, **kw)
    import torch

    report = {
        "date": time.strftime("%Y-%m-%d"),
        "data": {"frames": len(frames), "by_split": dict(Counter(f["meta"]["split"] for f in frames)), "by_os": dict(Counter(f["meta"]["os"] for f in frames)), "source": "Seed simulator computers (ubuntu/macos/windows themes) + demo web apps, 1280x800 / 1100x760 screenshots"},
        "models": load,
        "icon_memory": {"examples": len(memory.names), "distinct_names": len(set(memory.names)), "fit_on": str(args.icon_memory) if args.icon_memory else "tune split DOM labels of icon-only controls", "threshold": memory.threshold},
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.cuda.is_available(), "machine": platform.machine()},
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    full = results[next(iter(variants))]
    for split in ("tune", "test_app", "test_os"):
        if split in full:
            s = full[split]
            print(f"{split:<9} words R {s['words']['recall_fuzzy']} P {s['words']['precision_fuzzy']} | controls R {s['controls']['recall']} P {s['controls']['precision']} | target text {s['targeting_by_name']['text']['clicks_right_control']} (wrong {s['targeting_by_name']['text']['clicks_wrong_control']}) icon {s['targeting_by_name']['icon']['clicks_right_control']} field {s['targeting_by_name']['field']['clicks_right_control']} | windows {s['windows']['found']} title {s['windows']['title_correct']}")
    print("latency", full["latency_ms_per_frame"])


if __name__ == "__main__":
    main()
