# GR00T humanoid reference

Four actual rollouts of NVIDIA's pretrained **GR00T N1.6 G1 apple-to-plate policy**, with its upstream whole-body RL controller: **two completed the upstream task, two did not**. The default video is the first recorded failure; every trial is retained in [result.json](result.json). This small sample is a demonstration, not a reliable success-rate estimate or a matched comparison with TensorCode's different robot and task.

The high-level VLA sees ego RGB, language, and 43 joint angles. Its official low-level controller reads simulated joint state and floating-base orientation/angular velocity directly. This reference does **not** have TensorCode's separately implemented sensor-derived observation boundary.

The model weights are unchanged. Runtime adaptations use PyTorch SDPA instead of FlashAttention, direct BF16 loading, and two Python 3.12 dataclass fixes. Numerical equivalence to upstream FlashAttention was not separately established. Exact revisions, weight hashes, resources, and outcomes are in [result.json](result.json); all runtime edits are in [execution/](execution/).

## Reproduction

On an ARM64 CUDA 13 host with `uv`, `git-lfs`, `hf`, and user systemd:

```bash
bash execution/reproduce.sh
```

This downloads roughly 6.6 GB of model weights plus dependencies and assets into `/tmp/tensorcode-gr00t-reference`. The script records the setup and commands used, with a configurable scratch root; a second clean-environment installation has not been tested. The captured run reused an existing CUDA PyTorch installation read-only. The original executed runners and exact observed dependency versions are retained alongside setup patches.

Runs were limited to four CPU cores, a 48 GiB memory cgroup, zero cgroup swap, and an 18% PyTorch GPU allocation cap. A policy-request guard stopped work if available system RAM fell below 20%; the measured minimum was 68.8%. On GB10, GPU and system RAM share physical memory. Whole-host CPU headroom was not continuously logged during the captured trials; concurrent host load produced spikes. The reproduction script additionally applies hard four-core affinity and a reactive CPU/RAM pause guard.

See the [official benchmark guide](https://github.com/NVIDIA/Isaac-GR00T/blob/n1d6/examples/GR00T-WholeBodyControl/README.md) and [model](https://huggingface.co/nvidia/GR00T-N1.6-G1-PnPAppleToPlate).
