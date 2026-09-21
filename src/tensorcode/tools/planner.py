"""Trainable outcome prediction over explicitly supplied candidate plans."""
import math

import torch
from torch.nn import functional as F

from .._internal.pretrained import PretrainedTool
from .._internal.ranking import RankOperation, RankingObjective, RankingSession, bindings, from_foundation, normalize_config


class Planner(PretrainedTool):
    """Predict scalar outcomes and select the highest-scoring supplied plan.

    Does not execute plans or invent outcome feedback. Training can supervise one
    observed plan without assigning fabricated outcomes to unobserved plans.
    """

    def __init__(self, config):
        super().__init__(normalize_config(config))
        self.rank = RankOperation(self.config, task_key='goal', candidates_key='plans')
        self.objective = RankingObjective(self)

    from_foundation = classmethod(from_foundation)

    def forward(self, inputs, *, context=None):
        if context:
            raise ValueError('This tool does not accept context')
        return self.rank.receipt(inputs)

    predict = forward

    def new_session(self):
        return RankingSession(self)

    def loss(self, inputs, targets):
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
        return bindings(self)
