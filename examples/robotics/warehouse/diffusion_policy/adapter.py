"""Use exactly the recurrent BC sensor/walker/servo adapter."""

from ..learned_policy.adapter import LearnedController
from .model import PolicyRuntime


class DiffusionController(LearnedController):
    def __init__(self, *args, policy_directory, **kwargs):
        super().__init__(
            *args,
            policy_directory=policy_directory,
            runtime_factory=PolicyRuntime,
            **kwargs,
        )
