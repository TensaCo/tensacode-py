"""Typed Gymnasium environment transport without a task policy or inferred model.

Reset and step are explicit actions. Raw observations, rewards, termination, and
truncation remain separate evidence; a reward is never silently a belief or goal.
"""
from __future__ import annotations

from copy import deepcopy
from io import BytesIO
from typing import Any

from tensorcode.agent.plugin import Call, Capability, Param, Plugin
from tensorcode.outcomes import Receipt


class GymPlugin(Plugin):
    def __init__(self, environment: Any, *, name: str = "gym") -> None:
        super().__init__(name=name)
        self.environment = environment
        self.needs_reset = True
        self.closed = False
        self.sequence = 0
        self._observation: dict[str, Any] | None = None

    @classmethod
    def from_id(cls, environment_id: str):
        if not environment_id.strip():
            raise ValueError("gym requires an explicit registered environment ID")
        import gymnasium
        return cls(gymnasium.make(environment_id), name="gym:" + environment_id)

    def capabilities(self):
        return (
            Capability("reset", (Param("seed", "integer_or_none"),), effect_kind="write",
                       description="Explicitly reset the environment; seed is an integer or None."),
            Capability("step", (Param("action", "environment_action"),), effect_kind="write",
                       description="One action validated against this environment's action space."),
        )

    def observe(self) -> dict[str, Any]:
        return {"provenance": self.name, "sequence": self.sequence,
                "needs_reset": self.needs_reset, "closed": self.closed,
                "action_space": repr(self.environment.action_space),
                "observation_space": repr(self.environment.observation_space),
                "transition": deepcopy(self._observation),
                "limitations": ["Raw environment observations; no learned interpretation or policy"]}

    def execute(self, act: Call, *, key: str | None = None) -> Receipt:
        args = dict(act.args)
        if self.closed or act.plugin != self.name:
            return Receipt(act, "rejected", error="Environment is closed or provider does not match")
        if act.capability == "reset":
            if set(args) != {"seed"} or (args["seed"] is not None and
                    (type(args["seed"]) is not int or args["seed"] < 0)):
                return Receipt(act, "rejected", error="Reset requires a nonnegative integer seed or None")
        elif act.capability == "step":
            if self.needs_reset:
                return Receipt(act, "rejected", error="Explicit reset required before another step")
            try:
                valid = set(args) == {"action"} and self.environment.action_space.contains(args["action"])
            except Exception:
                valid = False
            if not valid:
                return Receipt(act, "rejected", error="Action is outside the environment's action space")
        else:
            return Receipt(act, "rejected", error="Unknown environment capability")
        try:
            if act.capability == "reset":
                observation, info = self.environment.reset(seed=args["seed"])
                transition = {"observation": observation, "info": info, "reward": None,
                              "terminated": False, "truncated": False, "operation": "reset"}
                self.needs_reset = False
            else:
                observation, reward, terminated, truncated, info = self.environment.step(args["action"])
                transition = {"observation": observation, "reward": reward, "info": info,
                              "terminated": bool(terminated), "truncated": bool(truncated), "operation": "step"}
                self.needs_reset = bool(terminated or truncated)
            self.sequence += 1
            self._observation = deepcopy(transition)
        except Exception as exc:
            # A timeout or malformed result can follow a mutation. Require an
            # explicit reset instead of retrying an action into unknown state.
            self.needs_reset = True
            self._observation = None
            return Receipt(act, "indeterminate", error=f"{type(exc).__name__}: {exc}")
        return Receipt(act, "applied", idempotency_key=key)

    def screenshot(self) -> bytes | None:
        if self.closed or self._observation is None or getattr(self.environment, "render_mode", None) != "rgb_array":
            return None
        from PIL import Image
        frame = self.environment.render()
        if frame is None:
            return None
        output = BytesIO()
        Image.fromarray(frame).save(output, format="PNG")
        return output.getvalue()

    def close(self) -> None:
        if not self.closed:
            self.environment.close()
            self.closed = True
