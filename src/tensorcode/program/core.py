"""Portable, bounded numerical programs. Artifacts contain data, never executable code."""
from __future__ import annotations

import copy
import itertools
import math
import struct
from dataclasses import dataclass
from typing import Any

OPS = {'add': 2, 'sub': 2, 'mul': 2, 'div': 2, 'neg': 1, 'relu': 1,
       'sigmoid': 1, 'gt': 2, 'ge': 2, 'lt': 2, 'le': 2, 'eq': 2,
       'ne': 2, 'select': 3, 'check': 1}
MODES = ('exact', 'relaxed', 'hybrid')


def float64(*, min=-1.7976931348623157e308, max=1.7976931348623157e308):
    return {'dtype': 'float64', 'min': min, 'max': max}


def float32(*, min=-3.4028234663852886e38, max=3.4028234663852886e38):
    return {'dtype': 'float32', 'min': min, 'max': max}


def int32(*, min=-2147483648, max=2147483647):
    return {'dtype': 'int32', 'min': min, 'max': max}


def boolean():
    return {'dtype': 'bool', 'min': 0, 'max': 1}


def check_domain(d):
    if not isinstance(d, dict) or set(d) != {'dtype', 'min', 'max'}:
        raise ValueError('domain requires dtype, min and max')
    if d['dtype'] not in ('float64', 'float32', 'int32', 'bool'):
        raise ValueError('unsupported dtype')
    lo, hi = d['min'], d['max']
    if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in (lo, hi)) or lo > hi:
        raise ValueError('invalid domain bounds')
    if d['dtype'] in ('int32', 'bool'):
        a, b = (-2147483648, 2147483647) if d['dtype'] == 'int32' else (0, 1)
        if lo != int(lo) or hi != int(hi) or lo < a or hi > b:
            raise ValueError('invalid integer domain')


def checked(x, d):
    if not isinstance(x, (int, float)) or not math.isfinite(x):
        raise ValueError('expected finite numeric value')
    x = float(x)
    if d['dtype'] in ('int32', 'bool') and x != math.trunc(x):
        raise ValueError('expected integer')
    if not d['min'] <= x <= d['max']:
        raise ValueError('value outside declared domain')
    if d['dtype'] == 'float32':
        x = struct.unpack('<f', struct.pack('<f', x))[0]
        if not math.isfinite(x) or not d['min'] <= x <= d['max']:
            raise ValueError('float32 rounding outside domain')
    return x


@dataclass(frozen=True)
class Value:
    builder: Any
    id: int

    def _op(self, op, *args): return self.builder.op(op, self, *args)
    def __add__(self, x): return self._op('add', x)
    def __radd__(self, x): return self.builder.op('add', x, self)
    def __sub__(self, x): return self._op('sub', x)
    def __rsub__(self, x): return self.builder.op('sub', x, self)
    def __mul__(self, x): return self._op('mul', x)
    def __rmul__(self, x): return self.builder.op('mul', x, self)
    def __truediv__(self, x): return self._op('div', x)
    def __neg__(self): return self._op('neg')
    def __gt__(self, x): return self._op('gt', x)
    def __ge__(self, x): return self._op('ge', x)
    def __lt__(self, x): return self._op('lt', x)
    def __le__(self, x): return self._op('le', x)
    def __bool__(self): raise TypeError('symbolic value: use select or compile source')


class Builder:
    def __init__(self, *, semantics='exact', dtype='float64', surrogate=None, temperature=1.0, max_nodes=100000):
        if semantics not in MODES or dtype not in ('float64', 'float32'):
            raise ValueError('invalid semantics or computation dtype')
        if semantics == 'hybrid' and surrogate != 'sigmoid':
            raise ValueError('hybrid requires explicit surrogate="sigmoid"')
        if surrogate not in (None, 'sigmoid') or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('invalid gradient policy')
        self.ir = dict(format='tensorcode.program', version=1, semantics=semantics,
                       dtype=dtype, surrogate=surrogate, temperature=temperature,
                       nodes=[], outputs={}, updates={})
        self.max_nodes = max_nodes
        self.source = None

    def _node(self, op, args=(), **attrs):
        if len(self.ir['nodes']) >= self.max_nodes: raise ValueError('program node budget exceeded')
        i = len(self.ir['nodes'])
        domain = attrs.pop('domain', float64() if self.ir['dtype'] == 'float64' else float32())
        n = dict(id=i, op=op, args=list(args), shape=[], domain=domain,
                 differentiability='discrete' if op in ('gt', 'ge', 'lt', 'le', 'eq', 'ne', 'gather', 'onehot') else 'piecewise',
                 source=self.source)
        n.update(attrs)
        self.ir['nodes'].append(n)
        return Value(self, i)

    def value(self, x):
        if isinstance(x, Value):
            if x.builder is not self: raise ValueError('value belongs to another builder')
            return x
        return self.constant(x)

    def constant(self, x):
        d = float64() if self.ir['dtype'] == 'float64' else float32()
        return self._node('const', value=checked(x, d), domain=d)

    def _named(self, op, name, domain, **attrs):
        check_domain(domain)
        if not isinstance(name, str) or not name or any(n.get('name') == name for n in self.ir['nodes']):
            raise ValueError('duplicate or empty binding name')
        return self._node(op, name=name, domain=copy.deepcopy(domain), **attrs)

    def input(self, name, domain=None): return self._named('input', name, domain or float64())
    def parameter(self, name, value, domain=None):
        d = domain or float64()
        if d['dtype'] not in ('float32', 'float64'): raise ValueError('parameters must be floating point')
        return self._named('parameter', name, d, value=checked(value, d))
    def state(self, name, initial, domain=None):
        d = domain or float64()
        return self._named('state', name, d, value=checked(initial, d))
    def update(self, state, value):
        state = self.value(state)
        if self.ir['nodes'][state.id]['op'] != 'state': raise ValueError('update target must be state')
        self.ir['updates'][str(state.id)] = self.value(value).id
    def op(self, op, *args):
        if op not in OPS or len(args) != OPS[op]: raise ValueError('unsupported operation or arity')
        domain = boolean() if op in ('gt', 'ge', 'lt', 'le', 'eq', 'ne') and self.ir['semantics'] != 'relaxed' else (float64() if self.ir['dtype'] == 'float64' else float32())
        return self._node(op, [self.value(a).id for a in args], domain=domain)
    def select(self, condition, yes, no): return self.op('select', condition, yes, no)
    def gather(self, values, index):
        if not values: raise ValueError('empty gather')
        return self._node('gather', [self.value(index).id] + [self.value(a).id for a in values])
    def one_hot(self, index, size):
        if type(size) is not int or not 1 <= size <= self.max_nodes: raise ValueError('invalid one-hot size')
        index=self._node('check',[self.value(index).id],domain=int32(min=0,max=size-1))
        return [self.op('eq',index,i) for i in range(size)]
    def gather_one_hot(self, values, selectors):
        if not values or len(values)!=len(selectors): raise ValueError('one-hot shape mismatch')
        index=self._node('onehot',[self.value(x).id for x in selectors],domain=int32(min=0,max=len(values)-1))
        return self.gather(values,index)
    def relu(self, x): return self.op('relu', x)
    def call(self, name, *args):
        if not isinstance(name, str) or not name: raise ValueError('call requires a binding name')
        return self._node('call', [self.value(a).id for a in args], name=name)
    def affine(self, values, weights, bias=0):
        if len(values) != len(weights): raise ValueError('affine shape mismatch')
        args=[self.value(bias).id]
        for x,w in zip(values,weights): args.extend([self.value(x).id,self.value(w).id])
        return self._node('affine',args)
    def sparse_affine(self, values, coefficients, bias=0):
        indices=sorted(coefficients)
        if any(type(i) is not int or not 0<=i<len(values) for i in indices): raise ValueError('invalid sparse affine index')
        return self.affine([values[i] for i in indices],[coefficients[i] for i in indices],bias)
    def scan(self, values, initial=0):
        result=[]; total=self.value(initial)
        for x in values:
            total=total+x; result.append(total)
        return result
    def include(self, program, inputs, *, prefix='region'):
        ir=program.graph()
        if ir['dtype'] != self.ir['dtype']: raise ValueError('composition dtype mismatch')
        expected={n['name'] for n in ir['nodes'] if n['op']=='input'}
        if set(inputs)!=expected: raise ValueError('composition signature mismatch')
        mapping={}
        for n in ir['nodes']:
            if n['op']=='input':
                value=self._node('check',[self.value(inputs[n['name']]).id],domain=n['domain'],source=n['source'])
            else:
                attrs={k:copy.deepcopy(v) for k,v in n.items() if k not in ('id','op','args')}
                if n['op'] in ('parameter','state'):
                    attrs['name']=prefix+'.'+n['name']
                    if any(x.get('name')==attrs['name'] for x in self.ir['nodes']): raise ValueError('duplicate composition binding')
                attrs['semantics']=n.get('semantics',ir['semantics'])
                attrs['temperature']=n.get('temperature',ir['temperature'])
                attrs['surrogate']=n.get('surrogate',ir['surrogate'])
                value=self._node(n['op'],[mapping[a].id for a in n['args']],**attrs)
            mapping[n['id']]=value
        for k,v in ir['updates'].items(): self.update(mapping[int(k)],mapping[v])
        return {k:mapping[v] for k,v in ir['outputs'].items()}

    def build(self, outputs):
        if not isinstance(outputs, dict): outputs = {'result': outputs}
        self.ir['outputs'] = {k: self.value(v).id for k, v in outputs.items()}
        return Program(self.ir)


def validate(ir):
    if not isinstance(ir, dict) or ir.get('format') != 'tensorcode.program' or ir.get('version') != 1 or type(ir.get('version')) is not int:
        raise ValueError('unsupported program artifact')
    if set(ir) != {'format','version','semantics','dtype','surrogate','temperature','nodes','outputs','updates'}: raise ValueError('unknown or missing artifact fields')
    Builder(semantics=ir['semantics'], dtype=ir['dtype'], surrogate=ir['surrogate'], temperature=ir['temperature'])
    nodes = ir['nodes']
    if not isinstance(nodes, list) or len(nodes) > 100000: raise ValueError('invalid node budget')
    names = set()
    for i, n in enumerate(nodes):
        if not isinstance(n,dict) or set(n)-{'id','op','args','shape','domain','differentiability','source','name','value','semantics','temperature','surrogate'}: raise ValueError('unknown node fields')
        if n.get('differentiability') not in ('discrete','piecewise'): raise ValueError('invalid differentiability metadata')
        op = n['op']; args = n['args']
        if not {'id','op','args','shape','domain','differentiability','source'}<=set(n): raise ValueError('missing node fields')
        if type(n['id']) is not int or n['id'] != i or n['shape'] != [] or not isinstance(args, list): raise ValueError('invalid scalar SSA node')
        if any(type(a) is not int or not 0 <= a < i for a in args): raise ValueError('non-topological SSA reference')
        check_domain(n['domain'])
        if n.get('semantics', ir['semantics']) not in MODES: raise ValueError('invalid node semantics')
        if n.get('surrogate',ir['surrogate']) not in (None,'sigmoid') or n.get('semantics',ir['semantics'])=='hybrid' and n.get('surrogate',ir['surrogate'])!='sigmoid': raise ValueError('hybrid nodes require explicit sigmoid surrogate')
        if not math.isfinite(n.get('temperature',ir['temperature'])) or n.get('temperature',ir['temperature']) <= 0: raise ValueError('invalid node temperature')
        arity = OPS.get(op, 0 if op in ('input', 'state', 'parameter', 'const') else None)
        if arity is not None and len(args) != arity: raise ValueError('invalid arity')
        if arity is None and not (op == 'call' or op == 'gather' and len(args) >= 2 or op == 'onehot' and len(args)>=1 or op == 'affine' and len(args)>=1 and len(args)%2==1): raise ValueError('unknown operation')
        if op=='call' and (not isinstance(n.get('name'),str) or not n['name']): raise ValueError('invalid call binding')
        if op in ('input', 'state', 'parameter'):
            name = n['name']
            if not isinstance(name, str) or not name or name in names: raise ValueError('duplicate binding')
            names.add(name)
        if op in ('state', 'parameter', 'const'): checked(n['value'], n['domain'])
        if op == 'affine' and n['domain']['dtype'] not in ('float32','float64'): raise ValueError('affine requires a floating computation dtype')
        if op == 'parameter' and n['domain']['dtype'] not in ('float32', 'float64'): raise ValueError('integer parameter')
    for mapping in (ir['outputs'], ir['updates']):
        if not isinstance(mapping, dict): raise ValueError('invalid outputs/updates')
        if any(not isinstance(k,str) for k in mapping): raise ValueError('output/update names must be strings')
        for v in mapping.values():
            if type(v) is not int or not 0 <= v < len(nodes): raise ValueError('invalid result reference')
    if not ir['outputs']: raise ValueError('program requires outputs')
    for k in ir['updates']:
        if not k.isdigit() or int(k) >= len(nodes) or nodes[int(k)]['op'] != 'state': raise ValueError('invalid state update')


def sigmoid(x):
    return 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))


class Program:
    def __init__(self, artifact):
        validate(artifact)
        self._ir = copy.deepcopy(artifact)

    def graph(self): return copy.deepcopy(self._ir)
    def to_json(self):
        import json
        return json.dumps(self._ir, allow_nan=False, separators=(',', ':'))
    @classmethod
    def from_json(cls, text):
        import json
        return cls(json.loads(text))

    def _execute(self, inputs, state=None, parameters=None, bindings=None, differentiation=False):
        ir = self._ir; nodes = ir['nodes']; cache = {}; edges = {}; order = []
        state = {} if state is None else state; parameters = {} if parameters is None else parameters; bindings = {} if bindings is None else bindings
        for mapping,kind,required in ((inputs,'input',True),(state,'state',False),(parameters,'parameter',False)):
            allowed={n['name']:n for n in nodes if n['op']==kind}
            if not isinstance(mapping,dict) or not set(mapping)<=set(allowed) or required and set(mapping)!=set(allowed): raise ValueError(kind+' signature mismatch')
            for name,value in mapping.items(): checked(value,allowed[name]['domain'])
        expected = {n['name'] for n in nodes if n['op'] == 'input'}
        if set(inputs) != expected: raise ValueError('input names must match the program signature')
        for n in nodes:
            if n['op'] == 'input': checked(inputs[n['name']], n['domain'])
        def calculate(i):
            if i in cache: return cache[i]
            n = nodes[i]; op = n['op']; ids = n['args']; deriv = []; mode = n.get('semantics', ir['semantics'])
            if op == 'input': y = inputs[n['name']]
            elif op == 'const': y = n['value']
            elif op == 'parameter': y = parameters.get(n['name'], n['value'])
            elif op == 'state': y = state.get(n['name'], n['value'])
            elif op == 'select':
                c = ev(ids[0])
                if mode == 'relaxed':
                    if not 0 <= c <= 1: raise ValueError('relaxed select requires a gate in [0,1]')
                    a, b = ev(ids[1]), ev(ids[2]); y = c*a+(1-c)*b
                    deriv = [(ids[0], a-b), (ids[1], c), (ids[2], 1-c)]
                else:
                    if c not in (0, 1): raise ValueError('exact select requires a boolean condition')
                    chosen = ids[1] if c else ids[2]; y = ev(chosen); deriv = [(chosen, 1)]
                    if mode == 'hybrid' and differentiation:
                        # A surrogate of selection needs both branch values, explicitly.
                        deriv.append((ids[0], ev(ids[1])-ev(ids[2])))
            elif op == 'gather':
                idx = ev(ids[0])
                if idx != int(idx) or not 0 <= idx < len(ids)-1: raise ValueError('gather index out of range')
                chosen = ids[int(idx)+1]; y = ev(chosen); deriv = [(chosen, 1)]
            else:
                a = [ev(j) for j in ids]
                if op == 'call':
                    if n['name'] not in bindings: raise ValueError('unbound explicit call: '+n['name'])
                    fn, vjp = bindings[n['name']]; y = fn(*a)
                    if vjp is not None and differentiation:
                        ds = vjp(a, y)
                        if len(ds) != len(a): raise ValueError('custom VJP arity')
                        deriv = list(zip(ids, ds))
                elif op == 'affine':
                    d=float64() if n['domain']['dtype']=='float64' else float32()
                    y=a[0]; deriv=[(ids[0],1)]
                    for j in range(1,len(a),2):
                        y=checked(y+checked(a[j]*a[j+1],d),d)
                        deriv.extend([(ids[j],a[j+1]),(ids[j+1],a[j])])
                elif op == 'onehot':
                    if any(x not in (0,1) for x in a) or sum(a)!=1: raise ValueError('expected exactly one active one-hot selector')
                    y=a.index(1)
                elif op == 'add': y = a[0]+a[1]; deriv = list(zip(ids, [1, 1]))
                elif op == 'sub': y = a[0]-a[1]; deriv = list(zip(ids, [1, -1]))
                elif op == 'mul': y = a[0]*a[1]; deriv = list(zip(ids, [a[1], a[0]]))
                elif op == 'div':
                    if a[1] == 0: raise ValueError('division by zero')
                    y = a[0]/a[1]
                    if differentiation: deriv = list(zip(ids, [1/a[1], -y/a[1]]))
                elif op == 'check': y = a[0]; deriv = [(ids[0], 1)]
                elif op == 'neg': y = -a[0]; deriv = [(ids[0], -1)]
                elif op == 'relu': y = max(0, a[0]); deriv = [(ids[0], float(a[0] > 0))]
                elif op == 'sigmoid': y = sigmoid(a[0]); deriv = [(ids[0], y*(1-y))]
                else:
                    hard = {'gt': lambda: a[0] > a[1], 'ge': lambda: a[0] >= a[1],
                            'lt': lambda: a[0] < a[1], 'le': lambda: a[0] <= a[1],
                            'eq': lambda: a[0] == a[1], 'ne': lambda: a[0] != a[1]}[op]()
                    y = float(hard)
                    if mode != 'exact':
                        t = n.get('temperature', ir['temperature']); delta = (a[0]-a[1])/t
                        if op in ('eq', 'ne'):
                            soft = math.exp(-delta*delta); g = -2*delta*soft/t if math.isfinite(delta) else 0
                            if op == 'ne': soft, g = 1-soft, -g
                        else:
                            sign = -1 if op in ('lt', 'le') else 1
                            soft = sigmoid(sign*delta); g = sign*soft*(1-soft)/t
                        if mode == 'relaxed': y = soft
                        deriv = [(ids[0], g), (ids[1], -g)]
            y = checked(y, n['domain']); cache[i] = y; edges[i] = deriv; order.append(i)
            return y
        def ev(root):
            # Explicit dependency stack avoids Python recursion limits for long scans.
            pending = [root]
            while pending:
                i = pending[-1]
                if i in cache:
                    pending.pop(); continue
                n = nodes[i]; ids = n['args']; deps = ids
                if n['op'] in ('select', 'gather'):
                    if ids[0] not in cache:
                        pending.append(ids[0]); continue
                    c = cache[ids[0]]
                    if n['op'] == 'select' and (n.get('semantics', ir['semantics']) == 'exact' or n.get('semantics', ir['semantics']) == 'hybrid' and not differentiation):
                        if c not in (0, 1): raise ValueError('exact select requires a boolean condition')
                        deps = [ids[1] if c else ids[2]]
                    elif n['op'] == 'gather':
                        if c != int(c) or not 0 <= c < len(ids)-1: raise ValueError('gather index out of range')
                        deps = [ids[int(c)+1]]
                missing = next((a for a in deps if a not in cache), None)
                if missing is not None:
                    pending.append(missing); continue
                calculate(i); pending.pop()
            return cache[root]
        outputs = {k: ev(v) for k, v in ir['outputs'].items()}
        next_state = {n['name']: checked(state.get(n['name'], n['value']), n['domain']) for n in nodes if n['op'] == 'state'}
        for k, v in ir['updates'].items():
            n = nodes[int(k)]; next_state[n['name']] = checked(ev(v), n['domain'])
        return outputs, next_state, (cache, edges, order)

    def run(self, inputs, **kwargs): return self._execute(inputs, **kwargs)[0]
    def step(self, inputs, **kwargs): return self._execute(inputs, **kwargs)[:2]
    def gradients(self, inputs, *, output='result', **kwargs):
        return self.linearize(inputs, **kwargs)[1][output]
    def linearize(self, inputs, **kwargs):
        outputs, _, (_, edges, order) = self._execute(inputs, differentiation=True, **kwargs)
        return outputs, {name: self._vjp(edges, order, name) for name in self._ir['outputs']}
    def _vjp(self, edges, order, output):
        if output not in self._ir['outputs']: raise ValueError('unknown output')
        adj = {self._ir['outputs'][output]: 1.0}
        for i in reversed(order):
            if not adj.get(i,0): continue
            n=self._ir['nodes'][i]
            if n['op']=='call' and n['args'] and not edges[i]: raise ValueError('differentiation through a call requires an explicit VJP')
            for j, d in edges[i]:
                value=adj.get(j, 0) + adj.get(i, 0)*d
                if not math.isfinite(value): raise ValueError('nonfinite gradient under selected policy')
                adj[j]=value
        return {n['name']: adj.get(n['id'], 0.0) for n in self._ir['nodes'] if n['op'] in ('input', 'parameter', 'state')}
    def verify(self, reference=None, *, cases=None, max_cases=100000, atol=0, rtol=0):
        reference = reference or getattr(self, '_reference', None)
        if reference is None: raise ValueError('provide an independent reference function')
        if cases is None:
            ins = [n for n in self._ir['nodes'] if n['op'] == 'input']
            total = 1; ranges = []
            for n in ins:
                d = n['domain']
                if d['dtype'] not in ('int32', 'bool'): raise ValueError('provide cases for continuous domains')
                total *= d['max']-d['min']+1
                if total > max_cases: raise ValueError('exhaustive verification exceeds budget')
                ranges.append(range(d['min'], d['max']+1))
            cases = (dict(zip([n['name'] for n in ins], row)) for row in itertools.product(*ranges))
            exhaustive = True
        else: exhaustive = False
        count = 0
        for case in cases:
            count += 1
            if count > max_cases: raise ValueError('verification exceeds budget')
            actual = self.run(case); expected = reference(**case)
            if isinstance(expected,(list,tuple)): expected = {str(i):v for i,v in enumerate(expected)}
            elif not isinstance(expected, dict): expected = {'result': expected}
            if set(expected) != set(actual) or any(not math.isclose(actual[k], expected[k], abs_tol=atol, rel_tol=rtol) for k in actual):
                return dict(passed=False, cases=count, exhaustive=exhaustive, counterexample=case, actual=actual, expected=expected)
        if count == 0: raise ValueError('verification requires cases')
        return dict(passed=True, cases=count, exhaustive=exhaustive)
    def schedule(self):
        levels = []; layers = []
        for n in self._ir['nodes']:
            level = 1+max((levels[a] for a in n['args']), default=-1); levels.append(level)
            while len(layers) <= level: layers.append([])
            layers[level].append(n['id'])
        return layers
    def optimize(self):
        ir = self.graph()
        effects=[]
        for n in ir['nodes']: effects.append(n['op']=='call' or any(effects[a] for a in n['args']))
        for n in ir['nodes']:
            # Pack multiply/add chains without reassociation or changing rounding points.
            if n['op']=='add' and not effects[n['id']]:
                acc,product=(ir['nodes'][i] for i in n['args'])
                full=float64() if n['domain']['dtype']=='float64' else float32()
                if product['op']=='mul' and n['domain']==full and product['domain']==full:
                    n['op']='affine';n['args']=[acc['id']]+product['args']
                    if acc['op']=='affine' and acc['domain']==full: n['args']=acc['args']+product['args']
            if n['op'] in ('input','state','parameter','const','call'): continue
            if all(ir['nodes'][a]['op']=='const' for a in n['args']):
                constants=[]
                for a in n['args']:
                    c=copy.deepcopy(ir['nodes'][a]);c['id']=len(constants);constants.append(c)
                target=copy.deepcopy(n);target['id']=len(constants);target['args']=list(range(len(constants)))
                temporary={**ir,'nodes':constants+[target],'outputs':{'result':target['id']},'updates':{}}
                try: value=Program(temporary).run({})['result']
                except (ValueError,OverflowError,ZeroDivisionError): continue
                n['op']='const';n['args']=[];n['value']=value
        live = set(ir['outputs'].values()) | set(ir['updates'].values())
        live |= {n['id'] for n in ir['nodes'] if n['op'] in ('input', 'state', 'parameter')}
        for n in reversed(ir['nodes']):
            if n['id'] in live: live.update(n['args'])
        mapping = {}; result = []
        for n in ir['nodes']:
            if n['id'] not in live: continue
            mapping[n['id']] = len(result); n['id'] = len(result); n['args'] = [mapping[a] for a in n['args']]; result.append(n)
        ir['nodes'] = result; ir['outputs'] = {k: mapping[v] for k,v in ir['outputs'].items()}
        ir['updates'] = {str(mapping[int(k)]): mapping[v] for k,v in ir['updates'].items()}
        return Program(ir)
    def freeze(self, names=None):
        ir=self.graph(); available={n['name'] for n in ir['nodes'] if n['op']=='parameter'}
        names=available if names is None else set(names)
        if not names<=available: raise ValueError('unknown parameter')
        for n in ir['nodes']:
            if n['op']=='parameter' and n['name'] in names:
                n['op']='const'; n.pop('name')
        return Program(ir)
    def to_tensor_module(self):
        from .tensor import ProgramModule
        return ProgramModule(self)
