import pytest
import torch
from tensorcode.ops.vec.latent import Latent, Space


def test_sequence_contract_masks_and_space():
    from tensorcode._internal.latent_ops import as_sequence
    s=Space('tokens',3,organization='sequence')
    x=torch.ones(2,3,requires_grad=True)
    value=Latent(x,s,mask=torch.tensor([True,False]))
    tensor,mask=as_sequence(value,s)
    assert tensor.shape==(1,2,3) and mask.tolist()==[[True,False]]
    assert tensor[0,1].tolist()==[0,0,0]
    tensor.sum().backward()
    assert x.grad[1].tolist()==[0,0,0]
    with pytest.raises(ValueError,match='incompatible'):
        as_sequence(value,Space('other',3,organization='sequence'))
    with pytest.raises(ValueError,match='valid'):
        as_sequence(Latent(x,s,mask=torch.zeros(2,dtype=torch.bool)),s)


def test_owned_operation_calls_trace_and_hooks():
    from tensorcode._internal.latent_ops import LatentOperation
    from tensorcode import trace
    class Scale(LatentOperation):
        def __init__(self):
            super().__init__({})
            self.weight=torch.nn.Parameter(torch.tensor(2.))
        def forward(self,value,*,context=None):
            return self.weight*value
    op=Scale(); seen=[]
    op.register_forward_hook(lambda *args:seen.append(True))
    with trace() as session:
        output=op(torch.tensor(3.))
    assert seen==[True] and len(session.calls)==1
    output.backward()
    assert op.weight.grad.item()==3
    assert op.operation_bindings()['operation'] is op
