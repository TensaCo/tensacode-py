"""AST compilation of a deliberately bounded, side-effect-free Python subset."""
import ast
import inspect
import textwrap
from .core import Builder, Value


def compile(source, *, inputs, semantics='exact', dtype='float64', surrogate=None,
            temperature=1.0, max_iterations=1024, max_nodes=100000):
    """Compile a function or source text without executing it or capturing closures.

    Supports scalar arithmetic, comparisons, conditional expressions/statements,
    local assignments, literal arrays/indexing, and statically bounded range loops.
    """
    reference = source if callable(source) else None
    if not isinstance(source, str): source = textwrap.dedent(inspect.getsource(source))
    tree = ast.parse(textwrap.dedent(source))
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError('expected one function definition')
    fn = tree.body[0]
    if fn.decorator_list or fn.args.vararg or fn.args.kwarg or fn.args.defaults or fn.args.kwonlyargs or fn.args.posonlyargs:
        raise ValueError('decorators, defaults and variadic signatures are unsupported')
    if [a.arg for a in fn.args.args] != list(inputs): raise ValueError('input signature mismatch')
    b = Builder(semantics=semantics, dtype=dtype, surrogate=surrogate,
                temperature=temperature, max_nodes=max_nodes)
    env = {k: b.input(k, v) for k,v in inputs.items()}
    binary = {ast.Add:'add', ast.Sub:'sub', ast.Mult:'mul', ast.Div:'div'}
    compare = {ast.Gt:'gt', ast.GtE:'ge', ast.Lt:'lt', ast.LtE:'le', ast.Eq:'eq', ast.NotEq:'ne'}
    budget = [0]
    def lower_expr(n, scope):
        b.source = {'language':'python', 'line':n.lineno, 'column':n.col_offset}
        if isinstance(n, ast.Constant) and type(n.value) in (int,float,bool):
            if type(n.value) is int and abs(n.value)>9007199254740991: raise ValueError('integer literal exceeds exact binary64 range')
            return b.constant(n.value)
        if isinstance(n, ast.Name) and n.id in scope: return scope[n.id]
        if isinstance(n, (ast.List, ast.Tuple)): return [expr(x,scope) for x in n.elts]
        if isinstance(n, ast.Subscript):
            values = expr(n.value,scope)
            if not isinstance(values,list): raise ValueError('indexing requires a fixed array')
            return b.gather(values,expr(n.slice,scope))
        if isinstance(n, ast.BinOp) and type(n.op) in binary: return b.op(binary[type(n.op)],expr(n.left,scope),expr(n.right,scope))
        if isinstance(n, ast.UnaryOp) and isinstance(n.op,ast.USub): return -expr(n.operand,scope)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op,ast.Not): return b.op('eq',expr(n.operand,scope),0)
        if isinstance(n, ast.Compare) and len(n.ops)==1 and type(n.ops[0]) in compare:
            return b.op(compare[type(n.ops[0])],expr(n.left,scope),expr(n.comparators[0],scope))
        if isinstance(n, ast.IfExp): return b.select(condition(n.test,scope),expr(n.body,scope),expr(n.orelse,scope))
        if isinstance(n, ast.Call) and isinstance(n.func,ast.Name) and n.func.id in ('relu','sigmoid') and len(n.args)==1 and not n.keywords:
            return b.op(n.func.id,expr(n.args[0],scope))
        raise ValueError(f'unsupported expression at line {n.lineno}: {ast.dump(n)}')
    def expr(n,scope):
        before=len(b.ir['nodes']); result=lower_expr(n,scope)
        if isinstance(result,Value) and result.id>=before:
            b.ir['nodes'][result.id]['source']={'language':'python','line':n.lineno,'column':n.col_offset}
        return result
    def condition(n,scope):
        v=expr(n,scope)
        return v if isinstance(n,ast.Compare) or isinstance(n,ast.UnaryOp) and isinstance(n.op,ast.Not) else b.op('ne',v,0)
    def merge(c,a,z):
        if isinstance(a,list) and isinstance(z,list) and len(a)==len(z): return [merge(c,x,y) for x,y in zip(a,z)]
        if isinstance(a,Value) and isinstance(z,Value): return b.select(c,a,z)
        raise ValueError('branches must have matching scalar/array structure')
    def block(stmts,scope):
        scope = dict(scope)
        for pos,s in enumerate(stmts):
            if isinstance(s,ast.Return): return expr(s.value,scope)
            if isinstance(s,ast.If):
                c = condition(s.test,scope)
                # Continuation lowering handles early returns and local mutation.
                return merge(c,block(s.body+stmts[pos+1:],scope),block(s.orelse+stmts[pos+1:],scope))
            if isinstance(s,ast.Assign) and len(s.targets)==1 and isinstance(s.targets[0],ast.Name): scope[s.targets[0].id]=expr(s.value,scope)
            elif isinstance(s,ast.AnnAssign) and isinstance(s.target,ast.Name) and s.value is not None: scope[s.target.id]=expr(s.value,scope)
            elif isinstance(s,ast.AugAssign) and isinstance(s.target,ast.Name) and type(s.op) in binary:
                scope[s.target.id]=b.op(binary[type(s.op)],scope[s.target.id],expr(s.value,scope))
            elif isinstance(s,ast.For) and isinstance(s.target,ast.Name) and not s.orelse:
                if any(isinstance(n,ast.Name) and isinstance(n.ctx,ast.Store) and n.id==s.target.id for stmt in s.body for n in ast.walk(stmt)):
                    raise ValueError('loop induction variable cannot be mutated in its body')
                call=s.iter
                if not isinstance(call,ast.Call) or not isinstance(call.func,ast.Name) or call.func.id!='range' or call.keywords:
                    raise ValueError('loops require literal range bounds')
                try: bounds=[ast.literal_eval(x) for x in call.args]
                except Exception as e: raise ValueError('loops require literal range bounds') from e
                if not 1 <= len(bounds) <= 3 or any(type(x) is not int for x in bounds): raise ValueError('invalid range')
                iterations=range(*bounds); budget[0]+=len(iterations)
                if budget[0]>max_iterations: raise ValueError('loop budget exceeded')
                expanded=[]
                for i in iterations:
                    assignment=ast.Assign(targets=[s.target],value=ast.Constant(value=i)); ast.copy_location(assignment,s); ast.copy_location(assignment.value,s)
                    expanded += [assignment]+s.body
                return block(expanded+stmts[pos+1:],scope)
            elif isinstance(s,ast.Pass): pass
            else: raise ValueError(f'unsupported statement at line {s.lineno}')
        raise ValueError('every path must return a value')
    result=block(fn.body,env)
    outputs={str(i):v for i,v in enumerate(result)} if isinstance(result,list) else result
    program = b.build(outputs)
    if semantics == 'exact':
        from .analysis import analyze
        if any(w['reason']=='integer arithmetic may lose precision' for w in analyze(program)['warnings']):
            raise ValueError('Python integer arithmetic may exceed exact binary64 range; narrow the input domains')
    program = program.optimize()
    program._reference = reference
    return program
