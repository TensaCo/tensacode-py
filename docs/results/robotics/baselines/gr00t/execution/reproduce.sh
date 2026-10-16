#!/usr/bin/env bash
# Reproduce this upstream reference separately from TensorCode's custom robot.
set -euo pipefail
bundle=$(cd "$(dirname "$0")" && pwd)
gr00t_work=${GR00T_WORK:-/tmp/tensorcode-gr00t-reference}
mkdir -p "$gr00t_work"
export GIT_LFS_SKIP_SMUDGE=1
if [ ! -d "$gr00t_work/Isaac-GR00T/.git" ]; then
  git clone --filter=blob:none --no-checkout https://github.com/NVIDIA/Isaac-GR00T "$gr00t_work/Isaac-GR00T"
fi
git -C "$gr00t_work/Isaac-GR00T" checkout 9b37aa1ce69c73c6d165233fa88128283bba4508
git -C "$gr00t_work/Isaac-GR00T" submodule update --init --depth 1 external_dependencies/GR00T-WholeBodyControl
wbc="$gr00t_work/Isaac-GR00T/external_dependencies/GR00T-WholeBodyControl"
git -C "$wbc" lfs pull --include='gr00t_wbc/sim2mujoco/**,gr00t_wbc/dexmg/**,gr00t_wbc/control/robot_model/**'
if [ ! -d "$gr00t_work/robosuite/.git" ]; then
  git clone --filter=blob:none --no-checkout https://github.com/xieleo5/robosuite "$gr00t_work/robosuite"
fi
git -C "$gr00t_work/robosuite" checkout b6aa4a53939ba872382eb32cb8948af0b5b79cdd
git -C "$gr00t_work/Isaac-GR00T" apply "$bundle/gr00t.patch"
git -C "$wbc" apply "$bundle/whole-body-control.patch"
uv venv --python 3.12 "$gr00t_work/venv"
py="$gr00t_work/venv/bin/python"
uv pip install --python "$py" 'torch==2.14.0+cu130' 'torchvision==0.29.0' --index-url https://download.pytorch.org/whl/cu130
# Exact observed versions; upstream training-only constraints differ from this inference overlay.
uv pip install --python "$py" --no-deps -r "$bundle/requirements.txt"
uv pip install --python "$py" --no-deps -e "$gr00t_work/Isaac-GR00T" -e "$wbc" -e "$wbc/gr00t_wbc/dexmg/gr00trobocasa" -e "$gr00t_work/robosuite"
hf download nvidia/GR00T-N1.6-G1-PnPAppleToPlate --revision 897d0313a190f46a2cccaeb34077752a0db4b0de --local-dir "$gr00t_work/checkpoint" --max-workers 2
"$py" - "$bundle" "$gr00t_work" <<'PY'
from pathlib import Path
import sys
bundle,work=map(Path,sys.argv[1:])
for name in ['run_episode.py','run_replication.py']:
    (work/name).write_text((bundle/name).read_text().replace('/tmp/gr00t-demo',str(work)))
PY
for runner in run_episode.py run_replication.py; do
  systemd-run --user --scope -p CPUQuota=400% -p MemoryMax=48G -p MemorySwapMax=0 \
    env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NO_ALBUMENTATIONS_UPDATE=1 \
    "$py" "$bundle/run_guarded.py" "$gr00t_work/$runner.resources.json" "$py" "$gr00t_work/$runner"
done
