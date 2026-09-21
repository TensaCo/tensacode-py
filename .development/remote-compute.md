# GB10 remote compute (2026-09-21)

All substantial training runs on the physically connected GB10 over SSH. The development host only edits/synchronizes source and retrieves small logs/reports. Work locally on `main`; no branches or worktrees are used. This file records environment setup, not a completed training result.

## Connection and dedicated workspace

```bash
ssh -o BatchMode=yes -o ConnectTimeout=10 gb10-direct hostname
# promaxgb10-4dfb
ssh gb10-direct
source /home/brandonin/tensorcode-runs/cognition-20260921/env.sh
python --version
```

Run root: `/home/brandonin/tensorcode-runs/cognition-20260921`.
Python: `/home/brandonin/tensorcode-runs/cognition-20260921/.venv/bin/python`.
Source: `$TENSORCODE_RUN_ROOT/repo`; logs: `$TENSORCODE_RUN_ROOT/logs`;
checkpoints/results: `$TENSORCODE_RUN_ROOT/artifacts`; public Hub cache:
`$TENSORCODE_RUN_ROOT/hf-cache`. The remote source copy has no `.git` directory.
No ancestor `AGENTS.md` was present at setup.

Initial resources: aarch64, NVIDIA GB10, 119 GiB total/115 GiB available unified
RAM, 814 GiB disk available, driver 580.126.09, CUDA driver capability 13.0.
The GPU was idle. `nvidia-smi` does not report dedicated VRAM capacity on this
unified-memory device; monitor system RAM as well as PyTorch allocation.

## Environment construction

An existing remote-only Python 3.12 environment at `/home/brandonin/ibm-1/.venv`
had working `torch==2.14.0+cu130`. Its CUDA matrix multiply returned the expected
64.0 value. The new environment was created and its site-packages independently
copied (`cp -a --reflink=auto`) from that environment. No original environment
files were changed; no credentials or private keys were copied. Dependencies
were then installed only into the new environment:

```bash
python3 -m venv /home/brandonin/tensorcode-runs/cognition-20260921/.venv
cp -a --reflink=auto /home/brandonin/ibm-1/.venv/lib/python3.12/site-packages/. \
  /home/brandonin/tensorcode-runs/cognition-20260921/.venv/lib/python3.12/site-packages/
/home/brandonin/tensorcode-runs/cognition-20260921/.venv/bin/python -m pip install \
  -e '/home/brandonin/tensorcode-runs/cognition-20260921/repo[tools,dev]' datasets pyarrow
```

Setup output is in `logs/setup.log`; exact installed versions are in
`logs/pip-freeze.txt` after setup. The preexisting Hugging Face cache contained
only Xet metadata (26 MiB), no cached foundation model weights.

## Source synchronization

Run from the local repository root. This captures tracked working-tree contents,
including uncommitted edits, but excludes untracked new files. Coordinate file
ownership and explicitly add newly authored source files to a subsequent sync.
It does not delete remote files or transfer credentials, `.git`, environments,
data, build outputs, or Node artifacts.

```bash
set -o pipefail
git ls-files -z | tar --exclude='.git' --exclude='.venv' --exclude='data' \
  --exclude='dist' --exclude='node_modules' --exclude='.env*' \
  --exclude='*.pem' --exclude='*.key' --null -T - -czf - | \
  ssh gb10-direct 'tar -xzf - -C /home/brandonin/tensorcode-runs/cognition-20260921/repo'
```

Initial synchronization used local main at
`b8369efd245a5f8d712d706c6a5f899d0b3f0531`. Later runs must record the actual
source commit and working-tree changes with their results.

## Run controls and inspection

`env.sh` sets `CUDA_VISIBLE_DEVICES=0`, `OMP_NUM_THREADS=8`, `MKL_NUM_THREADS=8`,
`TOKENIZERS_PARALLELISM=false`, `PYTHONUNBUFFERED=1`,
`HF_HUB_DISABLE_IMPLICIT_TOKEN=1`, and `HF_HUB_DISABLE_TELEMETRY=1`.
These are thread/concurrency settings, not hard memory limits. Begin with one
training process and no multiprocessing data-loader workers. Size actual batches
from a measured pilot, leaving at least 24 GiB available system RAM; do not change
unrelated processes or system GPU settings. Use a run-specific checkpoint directory.

```bash
ssh gb10-direct 'source /home/brandonin/tensorcode-runs/cognition-20260921/env.sh; python "$TENSORCODE_RUN_ROOT/smoke_cuda.py"'
ssh gb10-direct 'tail -40 /home/brandonin/tensorcode-runs/cognition-20260921/logs/setup.log'
ssh gb10-direct 'free -h; nvidia-smi; ps -u brandonin -o pid,etime,%cpu,%mem,args --sort=-%cpu | head -15'
```

For an assigned training command, run under `nohup` with a unique log and save
`$!` to the matching `.pid` file inside `logs`. Inspect that PID before sending a
signal. Record the exact command in the run report. Resume only using the training
script's documented checkpoint option and a checkpoint from that same run; setup
does not imply that arbitrary scripts support optimizer/RNG resume. A subprocess
exit status and final report must establish completion, not merely a PID exiting.

The tiny random-BERT smoke tests CUDA forward/backward and finite tensors. It
neither downloads pretrained weights nor demonstrates any learned cognitive skill.

## Verified setup outcome

CUDA tiny-BERT forward/backward passed with output `[2, 16, 64]`, finite loss
1.0, and 68,076,032 bytes peak CUDA allocation. Versions: Python 3.12.3,
PyTorch 2.14.0+cu130, Transformers 5.17.0, huggingface-hub 1.30.0,
datasets 5.0.1, pyarrow 25.0.1, NumPy 2.5.2, tensorcode 0.3.0a1.
Full smoke output is `logs/smoke-cuda.json` on GB10.

`pip check` reports `nvidia-cusparselt-cu13 0.8.1 is not supported on this
platform` for an inherited NVIDIA distribution. CUDA dense matrix multiply and
the transformer smoke nevertheless passed. Sparse CUDA functionality has not
been validated; do not treat the environment as universally validated or replace
the working CUDA build solely to silence this packaging warning.

The Hugging Face CLI entry point was recreated inside the isolated environment
with `python -m pip install --no-deps --force-reinstall huggingface-hub==1.30.0`.
