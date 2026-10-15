"""Bounded Python expression frontend. Parses AST; never executes supplied code."""

from __future__ import annotations

import ast

from .ir import AlgorithmBuilder


def compile(
    source, *, inputs, parameters=None, state=None, dtype="float64", max_unroll=1024
):
    """Compile one numerical function with static shapes and bounded literal loops.

    Supported: assignments, arithmetic, comparisons, conditional expressions,
    whitelisted tensor functions/methods, and literal-range for loops. State
    updates use update(state_variable, new_value). Both where branches are eager.
    """
    if type(max_unroll) is not int or not 1 <= max_unroll <= 10000:
        raise ValueError("invalid unroll limit")
    tree = ast.parse(source)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError("source must contain one function")
    fn = tree.body[0]
    if (
        fn.decorator_list
        or fn.args.defaults
        or fn.args.kw_defaults
        or fn.args.kwonlyargs
        or fn.args.vararg
        or fn.args.kwarg
        or fn.args.posonlyargs
    ):
        raise ValueError(
            "decorators and non-positional/default arguments are unsupported"
        )
    parameters = {} if parameters is None else parameters
    state = {} if state is None else state
    declared = list(inputs) + list(parameters) + list(state)
    if len(set(declared)) != len(declared) or {a.arg for a in fn.args.args} != set(
        declared
    ):
        raise ValueError("function argument signature mismatch")
    b = AlgorithmBuilder(name=fn.name, dtype=dtype)
    env = {k: b.input(k, v) for k, v in inputs.items()}
    env.update({k: b.parameter(k, v) for k, v in parameters.items()})
    env.update({k: b.state(k, v) for k, v in state.items()})
    binary = {
        ast.Add: "add",
        ast.Sub: "sub",
        ast.Mult: "mul",
        ast.Div: "div",
        ast.MatMult: "matmul",
    }
    comparisons = {
        ast.Gt: "gt",
        ast.GtE: "ge",
        ast.Lt: "lt",
        ast.LtE: "le",
        ast.Eq: "eq",
        ast.NotEq: "ne",
    }
    unary = {"relu", "tanh", "sigmoid", "sin", "cos", "exp", "sqrt", "log", "abs"}
    methods = unary | {
        "sum",
        "mean",
        "reshape",
        "transpose",
        "clip",
        "softmax",
        "slice",
    }
    budget = [0]

    def expr(node):
        if isinstance(node, ast.Constant) and type(node.value) in (
            int,
            float,
            bool,
            type(None),
        ):
            return node.value
        if isinstance(node, ast.Name) and node.id in env:
            return env[node.id]
        if isinstance(node, (ast.List, ast.Tuple)):
            return [expr(x) for x in node.elts]
        if isinstance(node, ast.Dict):
            if any(
                not isinstance(k, ast.Constant) or not isinstance(k.value, str)
                for k in node.keys
            ):
                raise ValueError("output keys must be literal strings")
            return {k.value: expr(v) for k, v in zip(node.keys, node.values)}
        if isinstance(node, ast.BinOp) and type(node.op) in binary:
            return b.op(binary[type(node.op)], expr(node.left), expr(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            x = expr(node.operand)
            return -x
        if (
            isinstance(node, ast.Compare)
            and len(node.ops) == 1
            and type(node.ops[0]) in comparisons
        ):
            return b.op(
                comparisons[type(node.ops[0])],
                expr(node.left),
                expr(node.comparators[0]),
            )
        if isinstance(node, ast.IfExp):
            return b.where(expr(node.test), expr(node.body), expr(node.orelse))
        if isinstance(node, ast.Call):
            if any(k.arg is None for k in node.keywords):
                raise ValueError("expanded kwargs unsupported")
            args = [expr(x) for x in node.args]
            kw = {k.arg: expr(k.value) for k in node.keywords}
            if isinstance(node.func, ast.Name):
                name = node.func.id
                if name in unary and len(args) == 1 and not kw:
                    return b.op(name, *args)
                if name == "where" and not kw:
                    return b.where(*args)
                if name == "concat":
                    return b.concat(*args, **kw)
                if name == "attention":
                    return b.attention(*args, **kw)
            if isinstance(node.func, ast.Attribute) and node.func.attr in methods:
                value = b.value(expr(node.func.value))
                return getattr(value, node.func.attr)(*args, **kw)
        raise ValueError(
            f"unsupported expression at line {getattr(node, 'lineno', '?')}: {type(node).__name__}"
        )

    def statements(body):
        for node in body:
            budget[0] += 1
            if budget[0] > 10000:
                raise ValueError("source instruction budget exceeded")
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
            ):
                env[node.targets[0].id] = expr(node.value)
            elif (
                isinstance(node, ast.AugAssign)
                and isinstance(node.target, ast.Name)
                and type(node.op) in binary
            ):
                env[node.target.id] = b.op(
                    binary[type(node.op)], env[node.target.id], expr(node.value)
                )
            elif isinstance(node, ast.Return):
                return expr(node.value)
            elif (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "update"
                and len(node.value.args) == 2
                and not node.value.keywords
            ):
                b.update(*(expr(x) for x in node.value.args))
            elif (
                isinstance(node, ast.For)
                and isinstance(node.target, ast.Name)
                and not node.orelse
            ):
                call = node.iter
                if (
                    not isinstance(call, ast.Call)
                    or not isinstance(call.func, ast.Name)
                    or call.func.id != "range"
                    or call.keywords
                ):
                    raise ValueError("loops require literal range")
                args = [expr(x) for x in call.args]
                if not 1 <= len(args) <= 3 or any(type(x) is not int for x in args):
                    raise ValueError("range arguments must be integer literals")
                r = range(*args)
                if len(r) > max_unroll:
                    raise ValueError("unroll budget exceeded")
                for i in r:
                    env[node.target.id] = i
                    result = statements(node.body)
                    if result is not None:
                        raise ValueError("return inside a loop is unsupported")
            else:
                raise ValueError(
                    f"unsupported statement at line {node.lineno}: {type(node).__name__}"
                )
        return None

    result = statements(fn.body)
    if result is None:
        raise ValueError("function must return outputs")
    return b.build(result)
