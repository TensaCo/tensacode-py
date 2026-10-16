"""Small recurrent BC policy, with serializable normalization and CPU inference."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .contract import ACTION_NAMES, SCHEMA, decode_action
from .data import sha256
from .geometry import GEOMETRY_FEATURE_NAMES as FEATURE_NAMES
from .geometry import extended_features


class GeometrySequencePolicy(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.hidden = hidden
        self.encoder = nn.Sequential(nn.Linear(len(FEATURE_NAMES), hidden), nn.Tanh())
        self.memory = nn.GRU(hidden, hidden, batch_first=True)
        self.output = nn.Linear(hidden, len(ACTION_NAMES))

    def forward(self, x, state=None):
        values, state = self.memory(self.encoder(x), state)
        return self.output(values), state


class GeometryRuntime:
    """Persistent recurrent state, reset once per episode; no teacher state access."""

    def __init__(self, directory):
        torch.set_num_threads(1)
        directory = Path(directory)
        self.config = json.loads((directory / "policy.json").read_text())
        if (
            self.config["schema"] != SCHEMA
            or self.config["feature_names"] != FEATURE_NAMES
            or self.config.get("action_names") != ACTION_NAMES
        ):
            raise ValueError("Policy observation contract mismatch")
        self.calibration = json.loads((directory / "kinematics.json").read_text())
        if self.config.get("kinematics_sha256") != sha256(
            directory / "kinematics.json"
        ):
            raise ValueError("Kinematic calibration hash mismatch")
        if (
            self.config.get("arm_output_representation")
            != "target_minus_current_own_encoder"
        ):
            raise ValueError("Unexpected geometry-policy target representation")
        self.model = GeometrySequencePolicy(self.config["hidden"])
        self.model.load_state_dict(
            torch.load(directory / "weights.pt", weights_only=True, map_location="cpu")
        )
        self.model.eval()
        self.mean = np.asarray(self.config["input_mean"], np.float32)
        self.scale = np.asarray(self.config["input_scale"], np.float32)
        self.ymean = np.asarray(self.config["action_mean"], np.float32)
        self.yscale = np.asarray(self.config["action_scale"], np.float32)
        for value, size in (
            (self.mean, len(FEATURE_NAMES)),
            (self.scale, len(FEATURE_NAMES)),
            (self.ymean, len(ACTION_NAMES)),
            (self.yscale, len(ACTION_NAMES)),
        ):
            if value.shape != (size,) or not np.isfinite(value).all():
                raise ValueError("Invalid policy normalization")
        if np.any(self.scale <= 0) or np.any(self.yscale <= 0):
            raise ValueError("Normalization scales must be positive")
        self.reset()

    def reset(self):
        self.state = None

    @torch.inference_mode()
    def step(self, observation):
        x = torch.from_numpy(
            (extended_features(self.calibration, [observation])[0] - self.mean)
            / self.scale
        )[None, None]
        prediction, self.state = self.model(x, self.state)
        prediction = prediction[0, 0].numpy()
        values = prediction * self.yscale + self.ymean
        values[3:10] += np.asarray(observation["joint_position"][22:29])
        values[-2:] = 1 / (1 + np.exp(-np.clip(prediction[-2:], -50, 50)))
        # Shared physical command limits, not an authored manipulation fallback.
        values[:3] = np.clip(values[:3], [-0.35, -0.35, -0.4], [0.35, 0.35, 0.4])
        # Teacher gripper actions are the two actuator modes 0 and .028m.
        # Decode at their midpoint so regression noise does not prevent opening.
        values[10] = 0.028 if values[10] >= 0.014 else 0.0
        return decode_action(values), values.tolist()
