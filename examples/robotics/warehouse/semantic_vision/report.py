"""Build a compact visual browser for every retained detector prediction."""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


def build(root):
    runs = []
    for path in sorted(root.glob("*/results.json")):
        summary = json.loads(path.read_text())
        predictions = json.loads(path.with_name("predictions.json").read_text())
        annotation_path = path.with_name("evaluation-amodal-boxes.json")
        if annotation_path.exists():
            annotations = {
                frame["path"]: frame["boxes"]
                for frame in json.loads(annotation_path.read_text())["frames"]
            }
            for row in predictions:
                if row["query"] not in ("a box", "a small cube"):
                    continue
                frame_dir = path.parent / row["frame"]
                canvas = Image.open(frame_dir / "sensor-rgb.png").convert("RGB")
                draw = ImageDraw.Draw(canvas)
                draw.rectangle((0, 0, canvas.width, 25), fill="#14202c")
                draw.text(
                    (8, 7),
                    "CYAN: neural proposals | AMBER: offline parcel labels (not model inputs)",
                    fill="white",
                )
                for detection in row["detections"]:
                    draw.rectangle(detection["box_xyxy"], outline="#00ffff", width=2)
                for truth in annotations.get(row["frame"], []):
                    draw.rectangle(truth["bbox_xyxy"], outline="#ffb534", width=2)
                canvas.save(frame_dir / "parcel-evaluation.jpg", quality=90)
        runs.append(
            {"path": path.parent.name, "summary": summary, "predictions": predictions}
        )
    payload = json.dumps(runs).replace("<", "\\u003c")
    template = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>TensorCode · Language-grounded vision</title>
<style>body{margin:0;background:#0b111b;color:#e7edf4;font:15px system-ui}main{max-width:1100px;margin:auto;padding:25px}h1{font-size:24px}p{color:#9daec0}select{background:#172334;color:white;border:1px solid #40536a;padding:10px;border-radius:7px;margin:5px}img{width:100%;max-height:72vh;object-fit:contain;background:#080d14}nav{display:flex;flex-wrap:wrap;gap:8px;margin:16px 0}.chip{padding:8px 12px;background:#18283b;border-radius:6px}#metrics{display:flex;gap:12px;flex-wrap:wrap;margin:15px 0}a{color:#78c7ff}small{color:#a8b9cb}figure{margin:0}</style>
<main><h1>Language-grounded vision</h1><p>Real pretrained Grounding DINO · external RGB-D replay · proposals, not robot commands</p>
<nav><span class="chip">RGB + query</span><span>→</span><span class="chip">Neural boxes</span><span>→</span><span class="chip">Measured depth / abstention</span></nav>
<label>Run <select id="run"></select></label><label>Query <select id="query"></select></label><label>View <select id="frame"></select></label>
<div id="metrics"></div><figure><img id="image" alt="Recorded RGB image with actual neural detection boxes"><figcaption id="caption"></figcaption></figure>
<p id="limits"></p><small>Depth points sample visible surfaces and may hit background. They are not object centers. Parcel labels are partial amodal projections; other scene boxes are unlabelled.</small>
<p><a id="evidence">Predictions + provenance</a> · <a id="labels">Independent parcel labels</a> · <a href="README.md">Study notes</a></p></main>
<script>const runs=PAYLOAD;const el=id=>document.getElementById(id);const fill=(id,values)=>{el(id).replaceChildren(...values.map(v=>{const o=document.createElement('option');o.value=v;o.textContent=v;return o}))};
fill('run',runs.map(r=>r.path));if(runs.some(r=>r.path==='evaluation-joint'))el('run').value='evaluation-joint';function changedRun(){const r=runs.find(r=>r.path===el('run').value);fill('query',[...new Set(r.predictions.map(p=>p.query))]);fill('frame',[...new Set(r.predictions.map(p=>p.frame))]);draw()}
function draw(){const r=runs.find(r=>r.path===el('run').value);const q=el('query').value,f=el('frame').value,p=r.predictions.find(p=>p.query===q&&p.frame===f);if(!p)return;const qi=[...new Set(r.predictions.map(p=>p.query))].indexOf(q);el('image').src=r.path+'/'+f+'/query-'+qi+'.jpg';el('caption').textContent=p.detections.length+' detection(s) · '+p.elapsed_s.toFixed(2)+' s CPU · '+f;el('evidence').href=r.path+'/results.json';el('labels').href=r.path+'/'+f+'/parcel-evaluation.jpg';const score=r.summary.partial_label_scoring;const negative=r.summary.per_query.find(p=>p.query==='an elephant');const metrics=[r.summary.frames+' recorded views',score?score.matched_at_iou_0_5+'/'+score.amodal_parcel_labels+' task parcels @ IoU .5':'No independent labels',negative.detections+' false elephant boxes',r.summary.depth_abstentions+' depth abstentions'];el('metrics').replaceChildren(...metrics.map(s=>{const x=document.createElement('span');x.className='chip';x.textContent=s;return x}));el('limits').textContent=q==='an elephant'?'Negative control: no elephant exists in this scene. Every reported elephant is a false positive.':'Cyan boxes are model outputs; missing task parcels remain failures. No thresholds were tuned on evaluation views.'}
el('run').onchange=changedRun;el('query').onchange=draw;el('frame').onchange=draw;changedRun();</script></html>"""
    if not runs:
        raise ValueError("No completed study runs")
    (root / "index.html").write_text(template.replace("PAYLOAD", payload))
    return root / "index.html"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    print(build(parser.parse_args().root))
