"""Trainable evidence-conditioned hypothesis ranking."""
import torch
from torch.nn import functional as F

from .._internal.pretrained import PretrainedTool
from .._internal.ranking import RankOperation, RankingObjective, RankingSession, bindings, from_foundation, normalize_config


class Investigator(PretrainedTool):
    """Own an encoder, shared workspace, and learned hypothesis scoring head.

    Candidate hypotheses and evidence are supplied inputs, not discovered facts.
    Probabilities are uncalibrated model scores. Construction uses random weights;
    use ``from_pretrained`` to load learned weights.
    """

    def __init__(self, config):
        super().__init__(normalize_config(config))
        self.rank = RankOperation(self.config, task_key='question', candidates_key='hypotheses')
        self.objective = RankingObjective(self)

    from_foundation = classmethod(from_foundation)

    def forward(self, inputs, *, context=None):
        if context:
            raise ValueError('This tool does not accept context')
        return self.rank.receipt(inputs, probabilities=True)

    predict = forward

    def new_session(self):
        return RankingSession(self)

    def loss(self, inputs, targets):
        logits = self.rank(inputs)
        if isinstance(targets, (list, tuple)) or isinstance(targets, torch.Tensor) and targets.ndim == 1:
            target = torch.as_tensor(targets, dtype=logits.dtype, device=logits.device)
            if target.shape != logits.shape or not torch.isfinite(target).all() or (target < 0).any() or not torch.isclose(target.sum(), target.new_tensor(1.0), atol=1e-5):
                raise ValueError('target distribution must be finite, nonnegative and sum to one')
            return -(target * logits.log_softmax(-1)).sum()
        if isinstance(targets, str):
            ids = [item['id'] for item in inputs['hypotheses']]
            if targets not in ids:
                raise ValueError('target must identify a supplied hypothesis')
            targets = ids.index(targets)
        if isinstance(targets, bool) or not isinstance(targets, int) or not 0 <= targets < logits.numel():
            raise ValueError('target must be a valid hypothesis index or ID')
        return F.cross_entropy(logits.unsqueeze(0), torch.tensor([targets], device=logits.device))

    @property
    def training_operation(self):
        return self.objective

    training_inputs_include_targets = True

    def operation_bindings(self):
        return bindings(self)
