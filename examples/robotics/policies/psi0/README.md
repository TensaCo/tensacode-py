# Released Ψ0 native humanoid reference

This adapter records the **public Ψ0 checkpoint from USC-PSI-Lab**, independently
of the locally trained Ψ0.1 policy and the unrelated Physical Intelligence π0.
It runs closed-loop on its trained SIMPLE tabletop task with a G1 and Dex3 hands.
It is not a warehouse-trained policy or a matched comparison to TensorCode's
pad-gripper workcell.

The high-level policy consumes head-camera RGB, task language and 32 state values
(28 measured hand/arm joint positions plus four previous torso/height commands).
It emits 30 × 36 action chunks; the native executor consumes the first 24 rows at
50 Hz. Actions specify fingers, arms, torso, base height, planar velocity and
heading; the upstream AMO whole-body tracker realizes locomotion. Simulator truth
is used for outcome judging and recorded diagnostics, not added to VLA inputs.
The idealized proprioception and native low-level controller are not claims of a
fully noise-realistic sensing pipeline.

Required external stack, reused without modifying it:

- Ψ0 `4f3720d45e102b36d7c3e9465ab8062274170518`.
- SIMPLE `803db7e4c1f0fef11008c9480ee75b2f9cc0ca36`.
- RRP's `rrp.harness.eval.video_psi0` recorder and `rrp.envs.simple` ARM/Isaac 5.1
  compatibility layer; source hashes accompany each evidence bundle.
- Released `USC-PSI-Lab/psi-model` checkpoint
  `psi0/simple-checkpoints/g1wholebodytabletopgrasp-v0.simple.flow1000.cosine.lr1.0e-04.b128.gpus8.2603181503/checkpoints/ckpt_40000/model.safetensors`.

On the prepared GB10, with the existing cached model and simulator environments:

```bash
PYTHONPATH=$HOME/work/rrp-src/src \
  $HOME/work/ext/venvs/psi/bin/python record.py --seed 0 --output /new/output/path
```

Use a bounded job: four cores, 48 GiB maximum memory, no swap, and 30-minute
maximum runtime. Monitor total host load and stop below 25% available RAM; pause
above 75% total CPU. Reserve the GPU slot against other inference/training jobs.
The output directory must be new. Every recorded outcome is retained, including
failures. Videos show the actual policy camera; they are not third-person renders
or replays of training actions.

Official implementation: https://github.com/physical-superintelligence-lab/Psi0
