"""Build a concise video gallery from measured robotics results.

Optional ``docs/results/robotics/transfer/showcase.json`` replaces the initial
cards. Every card requires id, title, subtitle, video and evidence. Asset paths
are relative to the robotics result directory; only existing files are shown.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

from architecture import architecture

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs/results/robotics"


def escape(value):
    return html.escape(str(value), quote=True)


def local_file(name):
    path = (OUT / name).resolve()
    if not path.is_relative_to(OUT.resolve()) or not path.is_file():
        raise ValueError(f"Missing or invalid showcase asset: {name}")
    return escape(name)


def default_cards():
    report = json.loads((OUT / "humanoid/report.json").read_text())
    measured = report["rollouts"]["alg2"]
    return [
        {
            "id": "walk-and-grasp",
            "title": "Walk. Reach. Grip. Lift.",
            "subtitle": "Earlier scripted demonstration · Unitree G1 · TensorCode ALG.2",
            "inputs": "Simulator ground-truth state",
            "method": "Authored manipulation · supplied walking policy",
            "video": "humanoid/humanoid-walk-grasp.mp4",
            "poster": "humanoid/humanoid-poster.png",
            "metrics": [
                {"value": f"{measured['walking_distance_m']:.2f} m", "label": "walked"},
                {
                    "value": f"{measured['max_lift'] * 100:.1f} cm",
                    "label": "object lift",
                },
                {
                    "value": f"{measured['held_above_10cm_seconds']:.2f} s",
                    "label": "held above 10 cm",
                },
            ],
            "evidence": "humanoid/report.json",
            "detail": "Pretrained Unitree locomotion with an authored manipulation controller. The walking network and joint servo execute through TensorCode. No new policy training; no visual in-context learning claim.",
        },
        {
            "id": "physical-contact",
            "title": "The grip makes the difference.",
            "subtitle": "Earlier scripted demonstration · closed versus open gripper",
            "inputs": "Simulator ground-truth state",
            "video": "humanoid/grip-control.mp4",
            "poster": "humanoid/control-poster.png",
            "metrics": [],
            "evidence": "humanoid/README.md",
            "detail": "A physical-contact control: both runs use native ONNX locomotion. Closing the jaws lifts the object; leaving them open does not. This is an ablation, not a comparison against a VLA or diffusion policy.",
        },
    ]


def render_card(card, number):
    video = local_file(card["video"])
    evidence = local_file(card["evidence"])
    still = (
        f'<a class="download" href="{local_file(card["still"])}">Ray-traced still ↗</a>'
        if card.get("still")
        else ""
    )
    poster = f' poster="{local_file(card["poster"])}"' if card.get("poster") else ""
    context = "".join(
        f'<span class="{field}">{escape(card[field])}</span>'
        for field in ("method", "inputs")
        if card.get(field)
    )
    context = f'<div class="context">{context}</div>' if context else ""
    clips = ""
    if card.get("clips"):
        options = "".join(
            f'<option value="{local_file(clip["video"])}"'
            + (" selected" if clip["video"] == card["video"] else "")
            + f">{escape(clip['label'])}</option>"
            for clip in card["clips"]
        )
        clips = f'<select class="clip-select" aria-label="Choose recorded episode">{options}</select>'
    metrics = "".join(
        f'<div class="metric"><strong>{escape(item["value"])}</strong><span>{escape(item["label"])}</span></div>'
        for item in card.get("metrics", [])
    )
    details = escape(
        card.get(
            "detail", "Recorded simulation. See the evidence for setup and limitations."
        )
    )
    kind = card.get("architecture", "legacy")
    diagram = architecture(kind, card["id"])
    return f'''<article class="demo" id="{escape(card["id"])}">
      <div class="demo-head"><div><span class="eyebrow">DEMO {number:02d}</span><h2>{escape(card["title"])}</h2><p>{escape(card["subtitle"])}</p></div><span class="simulation">SIMULATION</span></div>
      {context}
      <div class="screen"><video controls playsinline preload="metadata"{poster}><source src="{video}" type="video/mp4">Your browser cannot play this video. <a href="{video}">Download it.</a></video></div>
      <div class="demo-foot"><div class="metrics">{metrics}</div><div class="downloads">{clips}{still}<a class="download" href="{video}" download>Download video <span aria-hidden="true">↗</span></a></div></div>
      <section class="architecture"><div class="panel-heading"><h3>Controller architecture</h3><div class="legend"><span>Observed</span><span>Authored / estimated</span><span>Pretrained / frozen</span><span>Actuation</span></div></div><div class="diagram">{diagram}</div></section>
      <details><summary>How it works · evidence</summary><p>{details}</p><a href="{evidence}">Open measured results ↗</a></details>
    </article>'''


def main():
    manifest = OUT / "transfer/showcase.json"
    cards = (
        json.loads(manifest.read_text())["cards"]
        if manifest.exists()
        else default_cards()
    )
    if not cards:
        raise ValueError("The showcase requires at least one measured video")
    template = Path(__file__).with_name("report.html").read_text()
    values = {
        "EVIDENCE": local_file(cards[0]["evidence"]),
        "CARDS": "\n".join(render_card(card, i + 1) for i, card in enumerate(cards)),
        "NAV": "".join(
            f'<option value="{escape(card["id"])}">{i + 1:02d} · {escape(card["title"])}</option>'
            for i, card in enumerate(cards)
        ),
    }
    for key, value in values.items():
        template = template.replace("{{" + key + "}}", value)
    if "{{" in template:
        raise ValueError("Unresolved template field")
    (OUT / "index.html").write_text(template)
    (OUT / "kitting/index.html").write_text("""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="0;url=../index.html"><title>TensorCode robotics demos</title><p><a href="../index.html">Open humanoid robotics demos</a></p><p><a href="evidence.html">Archived kitting evidence</a></p></html>
""")
    print(OUT / "index.html")


if __name__ == "__main__":
    main()
