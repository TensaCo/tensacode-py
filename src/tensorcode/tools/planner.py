"""Owned plan generation and learned outcome prediction; text never executes."""
import math
import json

from .._internal.proposals import generate_proposals, proposal_loss

import torch
from torch.nn import functional as F

from .._internal.pretrained import PretrainedTool
from .._internal.ranking import RankOperation, RankingObjective, bindings, from_foundation, normalize_config
from .._internal.sessions.ranking import RankingSession
from .._internal.execution.planning import (
    PlanStep, ExecutablePlan, OutcomeExperience, ReplanRequest, PlanExecutionResult,
)

__all__ = ['Planner', 'PlanStep', 'ExecutablePlan', 'OutcomeExperience',
           'ReplanRequest', 'PlanExecutionResult']


class Planner(PretrainedTool):
    """Generate inert textual plans and rank predicted retrospective outcomes.

    A configured owned generator proposes candidate text when plans are omitted.
    Training can supervise one observed plan without assigning fabricated outcomes
    to unobserved alternatives. Predictions are not causal treatment estimates.
    """

    def __init__(self, config):
        config = dict(config)
        generator = None
        if config.get("generator") is not None:
            from .chatbot import Chatbot
            generator = Chatbot(config["generator"])
            config["generator"] = generator.configuration()
        super().__init__(normalize_config(config))
        self.generator = generator
        self.rank = RankOperation(self.config, task_key='goal', candidates_key='plans')
        self.objective = RankingObjective(self)

    from_foundation = classmethod(from_foundation)

    def forward(self, inputs, *, context=None):
        if context:
            raise ValueError('This tool does not accept context')
        if "plans" not in inputs:
            inputs = dict(inputs, plans=self.propose(inputs))
            if not inputs["plans"]:
                return {"selected_id": None, "candidates": [],
                        "evidence": json.loads(json.dumps(inputs.get("evidence", []))),
                        "abstained": True, "reason": "no_generated_plans"}
        receipt = self.rank.receipt(inputs)
        if any(not math.isfinite(item['predicted_score']) for item in receipt['candidates']):
            raise ValueError('planner predicted nonfinite scores; no plan selected')
        return receipt

    predict = forward

    def configuration(self):
        config = super().configuration()
        if self.generator is not None:
            config['generator'] = self.generator.configuration()
        return config

    def propose(self, inputs, *, count=3):
        """Generate inert plan text, retaining its full authored step text."""
        return generate_proposals(self.generator, inputs, task_key='goal',
                                  count=count, kind='plan')

    def generation_loss(self, inputs, targets):
        """Teacher-forced loss for the owned plan generator."""
        return proposal_loss(self.generator, inputs, targets, task_key='goal')

    @classmethod
    def from_foundations(cls, encoder_repo, generator_repo, *, encoder_revision=None,
                         generator_revision=None, local_files_only=False,
                         generator_options=None, **options):
        """Bootstrap owned foundation weights; ranking/workspace heads need training."""
        from .chatbot import Chatbot
        generator = Chatbot.from_foundation(generator_repo, revision=generator_revision,
            local_files_only=local_files_only, **(generator_options or {}))
        result = cls.from_foundation(encoder_repo, revision=encoder_revision,
            local_files_only=local_files_only, generator=generator.configuration(), **options)
        result.generator.load_state_dict(generator.state_dict())
        return result

    def new_session(self):
        """Create a ranking-history session sharing this tool's weights."""
        return RankingSession(self)

    def new_executor(self, *, actions, replan, max_steps):
        """Construct an executor without running actions or the replan policy.

        Calling the returned executor with state and an ExecutablePlan validates
        every step before any effect, then executes at most max_steps actions.
        """
        from .._internal.execution.planning import PlanExecutor
        return PlanExecutor(actions=actions, replan=replan, max_steps=max_steps)

    def loss(self, inputs, targets):
        """Outcome regression for one observed plan or one outcome per plan."""
        predictions = self.rank(inputs)
        if isinstance(targets, dict):
            ids = [item['id'] for item in inputs['plans']]
            if targets.get('candidate_id') not in ids:
                raise ValueError('candidate_id must identify a supplied plan')
            outcome = targets.get('outcome')
            if isinstance(outcome, bool) or not isinstance(outcome, (int, float)) or not math.isfinite(outcome):
                raise ValueError('outcome must be a finite number')
            return F.mse_loss(predictions[ids.index(targets['candidate_id'])], predictions.new_tensor(outcome))
        target = torch.as_tensor(targets, device=predictions.device, dtype=predictions.dtype)
        if target.shape != predictions.shape or not torch.isfinite(target).all():
            raise ValueError('targets must provide one finite observed outcome per plan')
        return F.mse_loss(predictions, target)

    @property
    def training_operation(self):
        return self.objective

    training_inputs_include_targets = True

    def operation_bindings(self):
        result = bindings(self)
        if self.generator is not None:
            result.update({"generator." + key: value for key, value in self.generator.operation_bindings().items()})
        return result
