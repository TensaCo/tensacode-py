# Robotics examples

| Example | Source and reproduction | Recorded report |
|---|---|---|
| Fixed-base actuator calibration | [Calibration guide](calibration/README.md) | [Results](../../docs/results/robotics/calibration/index.html) |
| G1 walking and physical gripping | [Humanoid guide](humanoid/README.md) | [Results](../../docs/results/robotics/humanoid/index.html) |

Each example owns its controller, report builder and HTML template. The two
report builders embed the shared `report.css` into standalone HTML. Evidence and
media stay beside each report under `docs/results/robotics/<example>/`.

Controllers are examples built on the general `tensorcode.algorithm` API;
robot-specific logic does not belong in the library core.
