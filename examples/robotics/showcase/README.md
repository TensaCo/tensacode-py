# Humanoid experiment dashboard

The public entry point is [robotics/index.html](../../../docs/results/robotics/index.html).
The earlier kitting URL redirects there; its original report remains at
[kitting/evidence.html](../../../docs/results/robotics/kitting/evidence.html).

Rebuild from the Python checkout:

```sh
python examples/robotics/showcase/build_report.py
```

The builder reads `docs/results/robotics/transfer/showcase.json` when present.
Otherwise it presents the existing measured humanoid walking and grip-control
recordings. The manifest has a `cards` array, ordered as displayed:

```json
{
  "cards": [
    {
      "id": "transfer",
      "title": "Replay a demonstrated reach.",
      "subtitle": "Unitree G1 · recorded physics simulation",
      "video": "transfer/comparison.mp4",
      "poster": "transfer/poster.png",
      "metrics": [],
      "inputs": "RGB-D · IMU · encoders · contact",
      "method": "One-demo retargeting; no gradient updates",
      "evidence": "transfer/report.json",
      "detail": "Describe the measured protocol and its limits here."
    }
  ]
}
```

All paths are relative to the robotics results directory. `poster`, `still` (a ray-traced presentation image), `metrics`,
`inputs`, `method`, and `detail` are optional. Metrics contain `value` and `label` strings and must
come from the referenced measurements. Optional `clips` entries contain `label`
and `video` paths, and expose all recorded episodes in a video selector. The builder rejects missing files and
paths outside the results directory. Videos remain separate files to avoid
inflating the landing page. Keep the page alongside its media when copying it.

Keep main-page copy short. Put experimental methodology and qualifications in
the expandable evidence section or linked report. Label authored procedures,
supplied models and learned behavior distinctly; rendering quality is not a
policy result.

The top experiment selector displays one dashboard panel at a time and supports
links to `#<card-id>`. Each panel has a paper-style SVG block diagram. Set
`architecture` to `sensor`, `transfer`, or `gr00t` for the corresponding implemented
controller; omitted values use the historical privileged-state diagram. Diagram
content lives in `architecture.py`; do not label authored RGB-D measurement
extraction as a learned image encoder.
