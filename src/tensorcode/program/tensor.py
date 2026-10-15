"""Optional PyTorch integration; importing the compiler does not import torch."""
import torch
from .core import Program


class ProgramModule(torch.nn.Module):
    """Owned scalar parameters with native state_dict/optimizer composition.

    Inputs and outputs are scalar tensors. Gradients implement the artifact's
    explicit policy; host calls require an explicitly supplied VJP binding.
    """
    def __init__(self, program: Program):
        super().__init__()
        self.program = program
        self._names = [n['name'] for n in program._ir['nodes'] if n['op']=='parameter']
        self.weights = [
            torch.nn.Parameter(torch.tensor(n['value'],dtype=torch.float64 if n['domain']['dtype']=='float64' else torch.float32))
            for n in program._ir['nodes'] if n['op']=='parameter']
        for i, weight in enumerate(self.weights): self.register_parameter(f'p{i}', weight)

    def forward(self, inputs, *, bindings=None):
        program=self.program
        names=list(inputs); tensors=list(inputs.values())+list(self.weights)
        if any(not isinstance(t,torch.Tensor) or t.ndim!=0 or not t.is_floating_point() for t in tensors):
            raise ValueError('program module expects scalar floating tensors')
        recording=torch.is_grad_enabled() and any(t.requires_grad for t in tensors)
        parameter_names=self._names
        output_names=list(program._ir['outputs'])
        class Execute(torch.autograd.Function):
            @staticmethod
            def forward(ctx,*xs):
                ctx.save_for_backward(*xs)
                ctx.values={k:t.item() for k,t in zip(names,xs)}
                ctx.params={k:t.item() for k,t in zip(parameter_names,xs[len(names):])}
                if recording: values,ctx.derivatives=program.linearize(ctx.values,parameters=ctx.params,bindings=bindings)
                else: values=program.run(ctx.values,parameters=ctx.params,bindings=bindings)
                dtype=torch.float64 if program._ir['dtype']=='float64' else torch.float32
                device=xs[0].device if xs else 'cpu'
                return tuple(torch.tensor(values[k],dtype=dtype,device=device) for k in output_names)
            @staticmethod
            def backward(ctx,*gs):
                xs=ctx.saved_tensors
                total={k:0.0 for k in names+parameter_names}
                for k,g in zip(output_names,gs):
                    if g is None: continue
                    deriv=ctx.derivatives[k]
                    for name in total: total[name]+=g.item()*deriv[name]
                return tuple(torch.full_like(t,total[k]) for k,t in zip(names+parameter_names,xs))
        outputs=Execute.apply(*tensors)
        return dict(zip(output_names,outputs))

    def artifact(self):
        ir=self.program.graph()
        params=dict(zip(self._names,self.weights))
        for n in ir['nodes']:
            if n['op']=='parameter': n['value']=params[n['name']].detach().item()
        return Program(ir)

ProgramModule.__module__ = "tensorcode.program"
