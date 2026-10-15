import json
import pytest
from tensorcode.program import Builder, Program, compile, int32, float32, float64


def test_branch_exhaustive_and_fractional():
    p=compile('def choose(a):\n return 10 if a > 4 else 12',inputs={'a':int32(min=-128,max=127)})
    assert p.verify(lambda a:10 if a>4 else 12)=={'passed':True,'cases':256,'exhaustive':True}
    with pytest.raises(ValueError): p.run({'a':4.5})
    p=compile('def choose(a):\n return 10 if a > 4 else 12',inputs={'a':float64()})
    assert p.run({'a':4.5})=={'result':10}
    assert Program.from_json(p.to_json()).run({'a':4.5})=={'result':10}


def test_loop_and_early_return():
    p=compile('''def f(x):
 s = 0
 for i in range(5):
  s += x + i
 if s > 20:
  return s
 return -s
''',inputs={'x':int32(min=-10,max=10)})
    assert p.verify(lambda x: (5*x+10) if 5*x+10>20 else -(5*x+10))['passed']


def test_lazy_selection_and_gather():
    b=Builder(); x=b.input('x'); p=b.build(b.select(x>0,1/x if False else b.op('div',1,x),7))
    assert p.run({'x':0})=={'result':7}
    b=Builder(); i=b.input('i',int32(min=0,max=2)); p=b.build(b.gather([3,5,8],i))
    assert p.run({'i':2})=={'result':8}
    with pytest.raises(ValueError): p.run({'i':3})


def test_simultaneous_state_and_failure_atomicity():
    b=Builder(); a=b.state('a',1,int32(min=0,max=9)); z=b.state('z',2,int32(min=0,max=9)); b.update(a,z); b.update(z,a)
    p=b.build(a+z); state={'a':3,'z':8}
    result,next_state=p.step({},state=state)
    assert result=={'result':11} and next_state=={'a':8,'z':3} and state=={'a':3,'z':8}


def test_gradient_modes_and_module_training():
    for mode in ('exact','relaxed','hybrid'):
        b=Builder(semantics=mode,surrogate='sigmoid' if mode=='hybrid' else None)
        x=b.input('x'); w=b.parameter('weight',2); p=b.build(b.select(x>0,w*x,0))
        assert p.gradients({'x':1})['weight']>0
        if mode!='relaxed': assert p.run({'x':1})['result']==2
    torch=pytest.importorskip('torch')
    b=Builder(); x=b.input('x'); w=b.parameter('weight',0); p=b.build(w*x)
    m=p.to_tensor_module(); opt=torch.optim.SGD(m.parameters(),lr=.1)
    for _ in range(20):
        opt.zero_grad(); loss=(m({'x':torch.tensor(1.,dtype=torch.float64)})['result']-3)**2; loss.backward(); opt.step()
    assert loss.item()<.01
    assert m.artifact().run({'x':1})['result']>2.9


def test_reject_malformed_and_unsupported():
    b=Builder(); x=b.input('x'); p=b.build(x+1)
    ir=p.graph(); ir['nodes'][-1]['args'][0]=999
    with pytest.raises(ValueError): Program(ir)
    with pytest.raises(ValueError): compile('def f(x):\n return print(x)',inputs={'x':float64()})
    with pytest.raises(ValueError): compile('def f(x):\n while x:\n  x -= 1\n return x',inputs={'x':float64()})
    with pytest.raises(ValueError): p.verify(lambda x:x+1)
    with pytest.raises(ValueError): Builder(semantics='hybrid')


def test_float32_rounding_custom_vjp_and_optimization():
    b=Builder(dtype='float32'); x=b.input('x',float32()); b.op('mul',x,100)
    y=b.call('square',x); p=b.build(y+1)
    bindings={'square':(lambda x:x*x,lambda args,y:[2*args[0]])}
    assert p.run({'x':3},bindings=bindings)=={'result':10}
    assert p.gradients({'x':3},bindings=bindings)['x']==6
    assert len(p.optimize().graph()['nodes'])<len(p.graph()['nodes'])
    assert p.schedule()


def test_composition_snapshot_and_hybrid_forward():
    from tensorcode.program import Runtime, analyze
    b=Builder(); a=b.input('a'); w=b.parameter('weight',2); region=b.build(w*a).freeze()
    c=Builder(semantics='hybrid',surrogate='sigmoid'); x=c.input('x'); r=c.parameter('correction',.5)
    p=c.build(c.include(region,{'a':x})['result']+r*x)
    assert p.run({'x':4})=={'result':10}
    assert p.gradients({'x':4})=={'x':2.5,'correction':4}
    assert analyze(p)['nodeCount']>0
    b=Builder(); s=b.state('s',0,int32(min=0,max=2)); b.update(s,s+1); runtime=Runtime(b.build(s))
    runtime.dispatch({}); restored=Runtime.restore(runtime.snapshot());restored.dispatch({})
    with pytest.raises(ValueError): restored.dispatch({})
    assert restored.state=={'s':2} and restored.steps==2
    b=Builder(semantics='hybrid',surrogate='sigmoid');x=b.input('x');p=b.build(b.select(x>0,b.op('div',1,x),7))
    assert p.run({'x':0})=={'result':7}
    with pytest.raises(ValueError): p.gradients({'x':0})


def test_operation_trainer():
    torch=pytest.importorskip('torch')
    from tensorcode.program.operation import ProgramOperation
    from tensorcode.training import Trainer
    from tensorcode import trace
    b=Builder();x=b.input('x');w=b.parameter('weight',0)
    op=ProgramOperation(b.build(w*x).graph());trainer=Trainer.from_ops({'op':op},lr=.1)
    with trace() as session:
        result=op({'x':torch.tensor(1.,dtype=torch.float64)})['result']
    session.supervise(result,torch.tensor(3.,dtype=torch.float64),loss='mse')
    losses=trainer.fit([session],epochs=20)
    assert losses[-1]<.01
    assert Program(op.artifact()).run({'x':1})['result']>2.9


def test_long_scan_and_precision_rejection():
    b=Builder();x=b.input('x');total=x
    for _ in range(2000):total=total+1
    assert b.build(total).run({'x':0})['result']==2000
    with pytest.raises(ValueError,match='precision|binary64'):
        compile('def f(x):\n return x*x',inputs={'x':int32()})


def test_one_hot_and_verified_replacement():
    from tensorcode.program import verified_replacement
    b=Builder();i=b.input('i',int32(min=0,max=2));p=b.build(b.gather_one_hot([3,5,8],b.one_hot(i,3)))
    assert p.run({'i':2})=={'result':8}
    assert verified_replacement(p,p.optimize()).run({'i':1})=={'result':5}
    b=Builder();p=b.build(b.gather_one_hot([3,5],[1,1]))
    with pytest.raises(ValueError):p.run({})


def test_prefix_scan_forward_call_and_tiny_division():
    b=Builder();x=b.input('x');p=b.build({str(i):v for i,v in enumerate(b.scan([x,1,2,3]))})
    assert p.run({'x':4})=={'0':4,'1':5,'2':7,'3':10}
    q=b.build(b.call('double',x));bindings={'double':(lambda x:2*x,None)}
    assert q.run({'x':2},bindings=bindings)=={'result':4}
    with pytest.raises(ValueError,match='VJP'):q.gradients({'x':2},bindings=bindings)
    b=Builder();a=b.input('a');z=b.input('z')
    assert b.build(a/z).run({'a':1e-300,'z':1e-300})=={'result':1}


def test_checkpoint(tmp_path):
    torch=pytest.importorskip('torch')
    from tensorcode.program.operation import ProgramOperation
    from tensorcode.training import Trainer
    from tensorcode import trace
    b=Builder();x=b.input('x');w=b.parameter('weight',0);artifact=b.build(w*x).graph()
    op=ProgramOperation(artifact);trainer=Trainer.from_ops({'op':op},lr=.1)
    with trace() as session: result=op({'x':torch.tensor(1.,dtype=torch.float64)})['result']
    session.supervise(result,torch.tensor(3.,dtype=torch.float64),loss='mse');trainer.step(session)
    trainer.save_checkpoint(tmp_path/'checkpoint')
    restored=ProgramOperation(artifact);continuation=Trainer.from_ops({'op':restored},lr=.1)
    continuation.load_checkpoint(tmp_path/'checkpoint')
    assert continuation.steps==1 and restored.artifact()==op.artifact()


def test_packed_affine_preserves_rounding_and_gradients():
    for dtype in ('float64','float32'):
        b=Builder(dtype=dtype);x=b.input('x');y=b.input('y');z=b.input('z');p=b.build(1+x*y+z*3);q=p.optimize()
        assert any(n['op']=='affine' for n in q.graph()['nodes'])
        assert len(q.graph()['nodes'])<len(p.graph()['nodes'])
        inputs={'x':1.234567,'y':7.654321,'z':-3}
        assert q.run(inputs)==p.run(inputs) and q.gradients(inputs)==p.gradients(inputs)


def test_imported_hybrid_policy():
    b=Builder(semantics='hybrid',surrogate='sigmoid',temperature=.5);a=b.input('a');region=b.build(b.select(a>0,1,0))
    outer=Builder();x=outer.input('x');p=outer.build(outer.include(region,{'a':x})['result'])
    assert p.run({'x':1})=={'result':1}
    assert p.gradients({'x':1})['x']==region.gradients({'a':1})['a']


def test_optimization_preserves_host_boundaries():
    b=Builder();x=b.input('x');effect=b.call('effect',x);p=b.build((b.constant(0)+x*x)+effect*1).optimize()
    calls=[]
    def forward(x): calls.append(x);return 1
    with pytest.raises(ValueError):p.run({'x':1e308},bindings={'effect':(forward,None)})
    assert calls==[]
