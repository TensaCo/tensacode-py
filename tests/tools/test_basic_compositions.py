from tensorcode.runtime import DecisionPipeline
from tensorcode.ops import vec
import torch


def test_decision_is_a_composition_of_public_operations():
    tool = DecisionPipeline(encode=vec.Transform(torch.nn.Identity()), decide=vec.Classify(torch.nn.Identity(), labels=('a','b')))
    assert tool(torch.tensor([0., 2.])).value == 'b'
