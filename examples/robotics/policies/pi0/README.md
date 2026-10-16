# π0 reference policy

This integration runs Physical Intelligence's released `pi0_aloha_sim` checkpoint
in its supported ALOHA cube-transfer simulator. It is a **bimanual arm reference**,
not a G1 checkpoint or a matched warehouse benchmark. The official DROID and
ALOHA action spaces cannot be relabeled as humanoid actions. G1 evaluation needs
an explicit embodiment adapter and adaptation evidence before it can be presented
as a supported policy.

Upstream: <https://github.com/Physical-Intelligence/openpi>, pinned revision
`215abfb217dbac7d5f1273282331b9b1866c0479`.
Weights: `gs://openpi-assets/checkpoints/pi0_aloha_sim` (12.01 GB JAX checkpoint).
The native task uses a rendered overhead RGB image and 14 joint/gripper state
values. No cube pose enters the policy. Simulation reward/contact state is used
only for evaluation. Proprioception and cameras are idealized simulation sensors.

Install the pinned upstream environment separately from TensorCode. Follow its
PyTorch conversion instructions if the local GPU cannot run the pinned JAX CUDA
runtime. Never patch a shared `transformers` installation: copy its files into an
isolated environment before applying upstream's required replacement modules.

`run.py` evaluates one seed, saves all executed actions and measured states, the
unaltered initial policy observation, a real simulation video, and outcome JSON.
It uses upstream's transforms and ten actions from each 50-action prediction,
matching the official ALOHA example's receding-horizon setting. It does not select
successful frames, retry a seed, or supply scripted recovery actions.

```bash
python run.py --checkpoint /path/to/pi0_aloha_sim_pytorch \
  --output /path/to/evidence --seed 0 --device cuda
```

Run this through the shared resource budget guard and coordinate GPU access with
other experiments. A completed prediction alone is not a successful task.

The native success predicate is reward 4: the cube touches the left gripper finger
without touching the table. It does **not** require sustained holding or release
by the right hand. `validate.py` recomputes this predicate from retained contact
pairs and checks every executed action against its saved prediction chunk.

`fetch.py` pins Google Cloud Storage object generations and verifies publisher
MD5 plus local SHA256. `convert.py` invokes the pinned official converter and
retains the norm statistics. `guard.py` limits four CPU cores/48 GiB, pauses above
75% global CPU use, and stops below 25% available memory. Its optional private MPS
connection requests 70% active CUDA threads; verify effective multiprocessor
visibility on the actual machine before claiming a GPU compute limit.
