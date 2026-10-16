"""Conditional DDPM with short sensor history and receding action chunks."""

from __future__ import annotations

import json
import math
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..learned_policy.contract import (
    ACTION_NAMES,
    FEATURE_NAMES,
    decode_action,
    features,
)

SCHEMA = "tensorcode.warehouse.diffusion.v1"
DEFAULT_ARCHITECTURE = {"history": 8, "horizon": 16, "hidden": 256, "blocks": 3}


class Denoiser(nn.Module):
    def __init__(self, history=8, horizon=16, hidden=256, blocks=3):
        super().__init__()
        self.history, self.horizon = history, horizon
        self.input = nn.Linear(
            history * len(FEATURE_NAMES) + horizon * len(ACTION_NAMES) + 32, hidden
        )
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden),
                    nn.Linear(hidden, hidden),
                    nn.SiLU(),
                    nn.Linear(hidden, hidden),
                )
                for _ in range(blocks)
            ]
        )
        self.output = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, horizon * len(ACTION_NAMES)),
        )
        self.register_buffer(
            "frequencies", torch.exp(torch.linspace(0, -math.log(10000), 16))
        )

    def forward(self, noisy_action, timestep, observation):
        angle = timestep[:, None].float() * self.frequencies[None]
        embedded = torch.cat([angle.sin(), angle.cos()], -1)
        hidden = self.input(
            torch.cat([noisy_action.flatten(1), observation.flatten(1), embedded], -1)
        )
        for block in self.blocks:
            hidden = hidden + block(hidden)
        return self.output(hidden).reshape(-1, self.horizon, len(ACTION_NAMES))


class Diffusion:
    def __init__(self, steps=32, device="cpu"):
        self.steps = steps
        time = torch.linspace(0, 1, steps + 1, dtype=torch.float64)
        cumulative = torch.cos((time + 0.008) / 1.008 * math.pi / 2).square()
        cumulative = cumulative / cumulative[0]
        self.beta = (
            (1 - cumulative[1:] / cumulative[:-1])
            .clamp(0.0001, 0.999)
            .float()
            .to(device)
        )
        self.alpha = 1 - self.beta
        self.cumulative = torch.cumprod(self.alpha, 0)
        self.previous = torch.cat([torch.ones(1, device=device), self.cumulative[:-1]])

    def noisy(self, clean, timestep, noise):
        a = self.cumulative[timestep, None, None]
        return a.sqrt() * clean + (1 - a).sqrt() * noise

    def loss(self, model, observations, actions, mask, generator=None):
        t = torch.randint(
            self.steps, (len(actions),), device=actions.device, generator=generator
        )
        noise = torch.randn(actions.shape, device=actions.device, generator=generator)
        prediction = model(self.noisy(actions, t, noise), t, observations)
        error = (prediction - noise).square().mean(-1)
        return (error * mask).sum() / mask.sum().clamp_min(1)

    @torch.inference_mode()
    def sample(self, model, observations, generator=None):
        shape = (len(observations), model.horizon, len(ACTION_NAMES))
        x = torch.randn(shape, device=observations.device, generator=generator)
        for step in reversed(range(self.steps)):
            t = torch.full((len(x),), step, dtype=torch.long, device=x.device)
            epsilon = model(x, t, observations)
            abar, previous, alpha, beta = (
                v[step] for v in (self.cumulative, self.previous, self.alpha, self.beta)
            )
            clean = ((x - (1 - abar).sqrt() * epsilon) / abar.sqrt()).clamp(-8, 8)
            x = (
                previous.sqrt() * beta / (1 - abar) * clean
                + alpha.sqrt() * (1 - previous) / (1 - abar) * x
            )
            if step:
                variance = beta * (1 - previous) / (1 - abar)
                x = x + variance.sqrt() * torch.randn(
                    shape, device=x.device, generator=generator
                )
        return x


class PolicyRuntime:
    """Same step/reset interface as BC; history contains only sensor features."""

    def __init__(self, directory):
        torch.set_num_threads(1)
        directory = Path(directory)
        config = json.loads((directory / "policy.json").read_text())
        if (
            config.get("schema") != SCHEMA
            or config.get("feature_names") != FEATURE_NAMES
            or config.get("action_names") != ACTION_NAMES
        ):
            raise ValueError("Incompatible diffusion sensor/action contract")
        self.model = Denoiser(**config["architecture"])
        self.model.load_state_dict(
            torch.load(directory / "weights.pt", weights_only=True, map_location="cpu")
        )
        self.model.eval()
        self.diffusion = Diffusion(config["diffusion_steps"])
        self.execute = int(config["execute"])
        if not 1 <= self.execute <= self.model.horizon:
            raise ValueError("Invalid execution horizon")
        self.seed = int(config["seed"])
        self.mean = np.asarray(config["input_mean"], dtype=np.float32)
        self.scale = np.asarray(config["input_scale"], dtype=np.float32)
        self.action_mean = np.asarray(config["action_mean"], dtype=np.float32)
        self.action_scale = np.asarray(config["action_scale"], dtype=np.float32)
        for value, size in (
            (self.mean, len(FEATURE_NAMES)),
            (self.scale, len(FEATURE_NAMES)),
            (self.action_mean, len(ACTION_NAMES)),
            (self.action_scale, len(ACTION_NAMES)),
        ):
            if value.shape != (size,) or not np.isfinite(value).all():
                raise ValueError("Invalid normalization")
        if np.any(self.scale <= 0) or np.any(self.action_scale <= 0):
            raise ValueError("Normalization scales must be positive")
        self.reset()

    def reset(self):
        self.history = deque(maxlen=self.model.history)
        self.pending = deque()
        self.generator = torch.Generator().manual_seed(self.seed)

    @torch.inference_mode()
    def step(self, observation):
        x = (features(observation) - self.mean) / self.scale
        if not self.history:
            self.history.extend([x.copy() for _ in range(self.model.history)])
        else:
            self.history.append(x)
        if not self.pending:
            inputs = torch.from_numpy(np.stack(self.history))[None]
            chunk = self.diffusion.sample(self.model, inputs, self.generator)[0].numpy()
            self.pending.extend(
                chunk[: self.execute] * self.action_scale + self.action_mean
            )
        values = self.pending.popleft().copy()
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite diffusion command")
        values[:3] = np.clip(values[:3], [-0.35, -0.35, -0.4], [0.35, 0.35, 0.4])
        values[10] = 0.028 if values[10] >= 0.014 else 0
        values[-2:] = values[-2:] >= 0.5
        return decode_action(values), values.tolist()
