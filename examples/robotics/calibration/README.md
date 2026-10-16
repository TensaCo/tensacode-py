# Fixed-base actuator calibration

`robotics_policy.py` compiles a 20-joint controller, learns an actuator residual
and compares ALG.0, ALG.1 and ALG.2 in MuJoCo. This is tracking on a fixed-base
mechanism, not walking. `build_robotics_report.py` replays the committed policy,
renders comparisons and assembles the HTML template using `../report.css`.

From the Python repository root:

```sh
python -m pip install -e '.[robotics]'
python examples/robotics/calibration/robotics_policy.py --help
python examples/robotics/calibration/build_robotics_report.py
```

[Recorded report](../../../docs/results/robotics/calibration/index.html).
The builder reads and writes that report's evidence directory. Existing numeric
records retain historical source paths and hashes; relocation is not a new run.
