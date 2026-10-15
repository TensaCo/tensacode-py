"""Transactional state and deterministic event replay."""
import copy
from .core import Program, checked


class Runtime:
    def __init__(self, program: Program, *, max_steps=100000):
        if type(max_steps) is not int or max_steps < 0: raise ValueError('invalid step budget')
        self.program=program
        self.max_steps=max_steps
        self.steps=0
        self.state={n['name']:n['value'] for n in program.graph()['nodes'] if n['op']=='state'}

    def dispatch(self, inputs, **kwargs):
        if self.steps >= self.max_steps: raise ValueError('execution step budget exceeded')
        outputs,state=self.program.step(inputs,state=self.state,**kwargs)
        self.state=state; self.steps+=1
        return outputs

    def snapshot(self):
        return {'format':'tensorcode.program-state','version':1,'program':self.program.graph(),
                'state':copy.deepcopy(self.state),'steps':self.steps,'maxSteps':self.max_steps}

    @classmethod
    def restore(cls, snapshot):
        if snapshot.get('format')!='tensorcode.program-state' or snapshot.get('version')!=1: raise ValueError('invalid state artifact')
        result=cls(Program(snapshot['program']),max_steps=snapshot['maxSteps'])
        if set(snapshot['state'])!=set(result.state): raise ValueError('state signature mismatch')
        if type(snapshot['steps']) is not int or not 0<=snapshot['steps']<=result.max_steps: raise ValueError('invalid step count')
        for n in result.program.graph()['nodes']:
            if n['op']=='state': checked(snapshot['state'][n['name']],n['domain'])
        result.state=copy.deepcopy(snapshot['state']); result.steps=snapshot['steps']
        return result

    def replay(self, events): return [self.dispatch(event) for event in events]
