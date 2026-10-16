"""An authored order/quality policy composed with a generic prerequisite circuit."""

from __future__ import annotations
import numpy as np
from tensorcode.algorithm import AlgorithmBuilder, compile_schedule, convert

TASKS = {
    **{f"inspect_{i}": [] for i in range(4)},
    **{f"quarantine_{i}": [f"inspect_{i}"] for i in range(4)},
    **{f"kit_{i}": [f"inspect_{j}" for j in range(4)] for i in range(3)},
    "seal": [f"kit_{i}" for i in range(3)] + [f"quarantine_{i}" for i in range(4)],
}


def compile_policy():
    b = AlgorithmBuilder(name="order_kitting")
    completed = b.input("completed", (12,))
    enabled = b.input("enabled", (12,))
    ready = b.include(
        compile_schedule(TASKS),
        {"completed": completed, "enabled": enabled},
        prefix="tasks",
    )["ready"]
    kind = b.input("kind", (4,))
    quality = b.input("quality", (4,))
    available = b.input("available", (4,))
    wanted = b.input("wanted")
    candidates = b.where(kind == wanted, b.where(quality >= 0.5, available, 0.0), 0.0)
    # Stable inventory order, not learned matching or recognition.
    score = candidates * b.constant([4.0, 3.0, 2.0, 1.0])
    return b.build({"ready": ready, "part_score": score})


class Policy:
    def __init__(self, substrate="alg0"):
        self.substrate = substrate
        self.algorithm = compile_policy()
        self.module = (
            None
            if substrate in ("native", "alg0")
            else convert(self.algorithm, substrate)
        )

    def __call__(self, inputs):
        if self.substrate == "native":
            done = np.asarray(inputs["completed"]) >= 0.5
            enabled = np.asarray(inputs["enabled"]) >= 0.5
            names = list(TASKS)
            ready = np.array(
                [
                    enabled[i]
                    and not done[i]
                    and all(done[names.index(p)] for p in TASKS[n])
                    for i, n in enumerate(names)
                ],
                float,
            )
            score = (
                (np.asarray(inputs["kind"]) == inputs["wanted"])
                * (np.asarray(inputs["quality"]) >= 0.5)
                * np.asarray(inputs["available"])
                * np.array([4.0, 3.0, 2.0, 1.0])
            )
            return {"ready": ready, "part_score": score}
        if self.substrate == "alg0":
            return self.algorithm.run(inputs)
        import torch

        with torch.no_grad():
            out = self.module(
                {k: torch.as_tensor(v, dtype=torch.float64) for k, v in inputs.items()}
            )
        return {k: v.detach().numpy() for k, v in out.items()}
