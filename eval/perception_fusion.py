"""Compare perception providers on the same frames: DOM, vision, and DOM+vision fused.

    PYTHONPATH=src:. python eval/perception_fusion.py --data DIR --cache DIR

The frames come from ``eval/vision_capture.py`` (screenshot + DOM scene + hit-tested word
ground truth). Configurations:

* ``dom``: the recorded DOM provider. It is also the ground truth, so its scores are an
  upper bound by construction, not a measurement. It is here to make that explicit.
* ``vision``: OCR + detector + CV rules on the screenshot alone.
* ``dom+vision``: fused, DOM preferred for structure.
* ``dom-degraded``: the DOM with accessible names removed from controls whose name is not
  visible as text (an app whose icon-only controls expose nothing) - the realistic case for
  native toolkits with partial accessibility.
* ``dom-degraded+vision``: fused. The question this answers: how much of what a thin
  accessibility tree loses can pixels put back?

Metrics per configuration: targeting by name the way agents do (``label_similarity``,
min 0.55, margin 0.08, click point must land inside the true control), control detection
recall/precision, word recall/precision, and how many fused items carry a conflict.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path[:0] = [str(Path(__file__).parents[1] / "src"), str(Path(__file__).parents[1])]

import numpy as np  # noqa: E402

from examples.browser_agents.browser import label_similarity  # noqa: E402
from examples.browser_agents.perception.fixture import FixtureProvider  # noqa: E402
from examples.browser_agents.perception.fusion import FusedProvider  # noqa: E402
from examples.browser_agents.perception.protocol import Target, inside, iou  # noqa: E402
from examples.browser_agents.perception.visual import from_pixel_scene  # noqa: E402
from examples.browser_agents.perception.web_dom import RecordedDomProvider  # noqa: E402
from examples.browser_agents.vision.icon_memory import IconMemory  # noqa: E402
from examples.browser_agents.vision.perceive import perceive  # noqa: E402
from eval.vision_perception import coarse, gt_controls, gt_words, load_frames, model_outputs, name_visible, norm  # noqa: E402

OUT = Path(__file__).parent / "results" / "perception_fusion.json"


def target_by_name(elements, query: str):
    scored = sorted(((label_similarity(query, e.name), e) for e in elements if e.name), key=lambda se: -se[0])
    if not scored or scored[0][0] < 0.55:
        return None
    if len(scored) > 1 and scored[0][0] - scored[1][0] < 0.08 and scored[0][1].name != scored[1][1].name:
        return None
    return scored[0][1]


def match_controls(elements, gold: list[dict]) -> list[tuple[int, int]]:
    pairs = []
    for gi, g in enumerate(gold):
        gb = tuple(g["box"])
        for pi, e in enumerate(elements):
            score = iou(e.box, gb) + (1.0 if inside(e.point, gb) else 0.0)
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


def match_words(texts, gold: list[dict]) -> int:
    """Ground-truth words covered by some text block that contains them at the right place."""
    hit = 0
    for g in gold:
        gb, gt = tuple(g["box"]), norm(g["text"])
        for t in texts:
            if iou(t.box, gb) >= 0.25 or inside((gb[0] + gb[2] // 2, gb[1] + gb[3] // 2), t.box):
                if gt and gt in norm(t.text):
                    hit += 1
                    break
    return hit


def evaluate(frames: list[dict], outputs: dict, memory: IconMemory | None) -> dict:
    from PIL import Image

    agg: dict = defaultdict(lambda: defaultdict(float))
    conflicts: dict = defaultdict(Counter)
    for f in frames:
        rgb = np.asarray(Image.open(f["png"]).convert("RGB"))
        cached = outputs[f["name"]]
        pixel_scene = perceive(rgb, ocr=lambda _: cached["words"], detector=lambda _: cached["icons"], icon_namer=memory)
        vision = FixtureProvider(from_pixel_scene(pixel_scene), "vision", 0.6)
        gw, gc = gt_words(f), gt_controls(f)
        dom = FixtureProvider(RecordedDomProvider(f["screen"]).perceive(Target()), "web-dom", 0.95)
        thin = FixtureProvider(RecordedDomProvider(f["screen"], drop_names_without_visible_text=True, words=gw).perceive(Target()), "web-dom", 0.95)
        configs = {
            "dom (= ground truth)": dom.perceive(Target()),
            "vision": vision.perceive(Target()),
            "dom+vision": FusedProvider([dom, vision]).perceive(Target()),
            "dom-degraded": thin.perceive(Target()),
            "dom-degraded+vision": FusedProvider([thin, vision]).perceive(Target()),
        }
        names = Counter(c["name"] for c in gc)
        for label, scene in configs.items():
            split = f["meta"]["split"]
            a = agg[(label, split)]
            a["frames"] += 1
            a["gt_controls"] += len(gc)
            a["elements"] += len(scene.elements)
            pairs = match_controls(scene.elements, gc)
            a["matched"] += len(pairs)
            for gi, pi in pairs:
                a[f"role_ok_{coarse(gc[gi]['role'])}"] += scene.elements[pi].role == coarse(gc[gi]["role"])
            a["gt_words"] += len(gw)
            a["words_found"] += match_words(scene.texts, gw)
            for c in gc:
                if not c["name"] or names[c["name"]] > 1 or len(c["name"]) > 40:
                    continue
                kind = "text" if name_visible(c, gw) else "icon" if coarse(c["role"]) == "button" and c["role"] != "combobox" else "field"
                hit = target_by_name(scene.elements, c["name"])
                a[f"{kind}_n"] += 1
                a[f"{kind}_ok"] += hit is not None and inside(hit.point, tuple(c["box"]))
                a[f"{kind}_wrong"] += hit is not None and not inside(hit.point, tuple(c["box"]))
            flagged = [x for x in (*scene.elements, *scene.texts) if x.conflicts]
            a["conflicts"] += len(flagged)
            a["sourced_by_vision"] += sum(1 for e in scene.elements if "vision" in e.sources and "web-dom" in e.sources)
            for x in flagged:
                conflicts[label][x.conflicts[0].field] += 1
    report: dict = {}
    for (label, split), a in agg.items():
        r = lambda n, d: round(a[n] / a[d], 3) if a[d] else None  # noqa: E731
        report.setdefault(label, {})[split] = {
            "frames": int(a["frames"]),
            "controls": {"gt": int(a["gt_controls"]), "predicted": int(a["elements"]), "recall": r("matched", "gt_controls"), "precision": r("matched", "elements")},
            "words_covered": r("words_found", "gt_words"),
            "targeting": {k: {"queries": int(a[f"{k}_n"]), "right": r(f"{k}_ok", f"{k}_n"), "wrong": r(f"{k}_wrong", f"{k}_n")} for k in ("text", "icon", "field")},
            "items_with_a_conflict": int(a["conflicts"]),
            "elements_seen_by_both_sources": int(a["sourced_by_vision"]),
        }
    for label in report:
        report[label]["conflict_fields"] = dict(conflicts[label])
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--icon-memory", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    frames = load_frames(args.data)
    outputs, load = model_outputs(frames, args.cache)
    memory = IconMemory.load(args.icon_memory) if args.icon_memory else None
    t0 = time.perf_counter()
    report = evaluate(frames, outputs, memory)
    out = {
        "date": time.strftime("%Y-%m-%d"),
        "frames": len(frames),
        "by_split": dict(Counter(f["meta"]["split"] for f in frames)),
        "icon_memory": {"examples": len(memory.names)} if memory else None,
        "models": load,
        "environment": {"python": platform.python_version(), "machine": platform.machine()},
        "seconds": round(time.perf_counter() - t0, 1),
        "providers": report,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    for label, splits in report.items():
        for split in ("tune", "test_app", "test_os"):
            if split in splits:
                s = splits[split]
                t = s["targeting"]
                print(f"{label:<22} {split:<9} ctrl R/P {s['controls']['recall']}/{s['controls']['precision']}  words {s['words_covered']}  "
                      f"target text {t['text']['right']}(w {t['text']['wrong']}) icon {t['icon']['right']} field {t['field']['right']}  conflicts {s['items_with_a_conflict']}")
        print()
    _ = statistics


if __name__ == "__main__":
    main()
