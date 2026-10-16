"""Audit the first noisy-action projection without changing a checkpoint."""

import argparse
import json
import sys
from pathlib import Path

import torch

parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--policy", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
sys.path.insert(0, str(args.source.resolve()))
from examples.robotics.warehouse.diffusion_policy.model import PolicyRuntime
from examples.robotics.warehouse.learned_policy.contract import (
    ACTION_NAMES,
    FEATURE_NAMES,
)
from examples.robotics.warehouse.learned_policy.data import sha256

torch.set_num_threads(1)
model = PolicyRuntime(args.policy).model
dimensions = model.horizon * len(ACTION_NAMES)
projection = model.input.weight[:, :dimensions].detach().double()
_, _, vectors = torch.linalg.svd(projection, full_matrices=True)
rank = int(torch.linalg.matrix_rank(projection))
perturbation_error = None
if rank < dimensions:
    null = vectors[-1].float().reshape(1, model.horizon, len(ACTION_NAMES))
    condition = torch.zeros(1, model.history, len(FEATURE_NAMES))
    timestep = torch.tensor([31])
    with torch.inference_mode():
        perturbation_error = float(
            (
                model(null, timestep, condition)
                - model(torch.zeros_like(null), timestep, condition)
            )
            .abs()
            .max()
        )
result = {
    "noisy_action_dimensions": dimensions,
    "first_layer_width": projection.shape[0],
    "numerical_rank": rank,
    "unobserved_gaussian_dimensions": dimensions - rank,
    "epsilon_mse_lower_bound_full_unmasked_chunks": (dimensions - rank) / dimensions,
    "unit_null_perturbation_prediction_max_change": perturbation_error,
    "weights_sha256": sha256(args.policy / "weights.pt"),
    "config_sha256": sha256(args.policy / "policy.json"),
    "scope": "First-layer information bottleneck; does not prove the cause of each physical failure.",
    "model_unchanged": True,
}
args.output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
