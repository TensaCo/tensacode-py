# Unitree G1 simulation components

Shared physical model, frozen locomotion, joint servo and inverse kinematics for
[robotics lab sessions](../../../docs/results/robotics/index.html). Robot-specific
implementation stays outside the general TensorCode library.

- `model.py`: scene construction, proprioceptive observations, frozen walker
  conversion, servo compilation and inverse kinematics.
- `fetch_assets.py`: acquire the pinned robot, meshes, configuration and weights.
- `budget.py`: bound CPU/RAM usage and pause on sampled machine pressure.

Install `pip install -e '.[robotics]'` from the Python repository, then run:

```bash
python examples/robotics/g1/fetch_assets.py \
  --output /tmp/tensorcode-humanoid-assets/unitree_rl_mjlab
```

Assets come from [Unitree RL mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab),
revision `1425b15f73bd4095f0df53709d7c389c3eb9e790`.
The ONNX walker SHA-256 is
`2a66ca6336eadb3c0b34b557763f3e06d01ff8fcf6260dd4cedbd69d6093fc28`.
Upstream assets are Apache-2.0 licensed; see [UNITREE-LICENCE](UNITREE-LICENCE).
The controller uses supplied pretrained walking weights, not a walker trained here.

[Sensor-driven session reproduction](../transfer/README.md) uses these components.
The resource guard samples load reactively; it cannot guarantee global headroom
against unrelated processes between samples.
