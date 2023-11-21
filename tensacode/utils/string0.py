import inspect
from types import LambdaType, ModuleType
from typing import Callable
from textwrap import dedent, indent
import attr
from jinja2 import Template
from loguru import logger
from pydantic import BaseModel
from tensacode.utils.string0 import render_invocation

INDENTATION = "    "


def render_invocation(fn: str or Callable, /, args=None, kwargs=None, result=None):
    if callable(fn):
        fn = fn.__name__

    args = args or []
    kwargs = kwargs or {}

    args_str = ", ".join(map(str, args))
    kwargs_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())

    result_str = f" -> {result}" if result is not None else ""

    if args_str and kwargs_str:
        return f"{fn}({args_str}, {kwargs_str}){result_str}"
    elif args_str:
        return f"{fn}({args_str}){result_str}"
    elif kwargs_str:
        return f"{fn}({kwargs_str}){result_str}"
    else:
        return f"{fn}(){result_str}"


def _render_annotation(annotation):
    """
    Formats the annotation, using the qualified name for custom types,
    the simple name for built-in types, and recursively including arguments for generic types.

    Args:
    annotation: The annotation to format.

    Returns:
    str: The formatted annotation.
    """
    # If the annotation is a built-in type (like int, str), return its name
    if getattr(annotation, "__module__", None) == "builtins":
        return annotation.__name__
    # If the annotation is a generic type with arguments (like List[int])
    elif hasattr(annotation, "__origin__") and hasattr(annotation, "__args__"):
        args_formatted = ", ".join(
            [_render_annotation(arg) for arg in annotation.__args__]
        )
        return f"{_render_annotation(annotation.__origin__)}[{args_formatted}]"
    # For custom types, use the qualified name
    else:
        return annotation.__qualname__


def render_block(head, body) -> str:
    return f"{head}:\n{indent(body, INDENTATION)}"


def render_function(func: Callable) -> str:
    '''
    Generates a string representation of the given function including its name, parameters,
    annotations, default values, return type, and docstring, handling various parameter types.

    Args:
        func (Callable): The function to represent.

    Returns:
        str: The string representation of the function.

    Example usage:
        >>> def new_test_function(x: int, y: list[str], z: dict[str, float]) -> bool:
                """This is a new test function with complex annotations."""
                return True
            # Get the representation
        >>> function_representation(new_test_function)
        ... def new_test_function(x: int, y: list[str], z: dict[str, float]) -> bool:
                """This is a new test function with complex annotations."""
                ...

    '''
    # Get the function's name
    func_name = func.__name__

    # Get the signature of the function
    signature = inspect.signature(func)

    # Construct parameter string with annotations, default values, and special types
    params = []
    positional_only_separator_added = False
    for name, param in signature.parameters.items():
        # Handle variable positional (*args) and keyword (**kwargs) parameters
        if param.kind == param.VAR_POSITIONAL:
            params.append(f"*{name}")
        elif param.kind == param.VAR_KEYWORD:
            params.append(f"**{name}")
        else:
            # Include formatted annotation if available
            annotation = (
                _render_annotation(param.annotation)
                if param.annotation is not param.empty
                else ""
            )
            param_str = f"{name}: {annotation}" if annotation else name
            # Add default value if present
            if param.default is not inspect.Parameter.empty:
                param_str += f"={param.default}"
            params.append(param_str)

        # Add '/' after positional-only parameters
        if param.kind == param.POSITIONAL_ONLY and not positional_only_separator_added:
            params.append("/")
            positional_only_separator_added = True

    # Remove the last '/' if it's at the end of the list
    if params and params[-1] == "/":
        params.pop()

    params_str = ", ".join(params)

    # Get the return type
    return_annotation = signature.return_annotation
    return_annotation_str = (
        f" -> {_render_annotation(return_annotation)}"
        if return_annotation is not inspect.Signature.empty
        else ""
    )

    # Get the docstring
    docstring = inspect.getdoc(func)
    docstring_str = f'"""{docstring}"""' if docstring else ""

    # Combine everything into the final string

    return render_block(
        head=f"def {func_name}({params_str}){return_annotation_str}",
        body=f"{docstring_str}\n...",
    )


def render_lambda(lmbda: LambdaType) -> str:
    return Template(
        dedent(
            """\
            lambda {{ args }}: {{ body }}
            """
        )
    ).render(
        args=", ".join(lmbda.__code__.co_varnames),
        body=render_atomic_object(lmbda.__code__.co_consts[0]),
    )


def render_atomic_object(obj: object) -> str:
    if isinstance(obj, LambdaType):
        return render_lambda(obj)
    elif isinstance(obj, BaseModel):
        return render_invocation(
            obj.__class__,
            kwargs={k: render_atomic_object(v) for k, v in obj.model_dump().items()},
        )
    else:
        return str(obj)


def render_member(container: object, key: str, val: object = None) -> str:
    if inspect.isfunction(val):
        return render_function(val)
    elif inspect.isclass(val):
        return render_class(val)
    elif inspect.ismodule(val):
        return render_module(val)
    else:
        annotation: str = None
        if hasattr(container, "__annotations__"):
            annotation = container.__annotations__.get(key, None)
            if annotation is not None:
                annotation = _render_annotation(annotation)
        match annotation, val:
            case None, None:
                logger.warning("Cannot render member without annotation or value")
                return f"{key}"
            case None, _:
                return f"{key} = {render_atomic_object(val)}"
            case _, None:
                return f"{key}: {annotation}"
            case _, _:
                return f"{key}: {annotation} = {render_atomic_object(val)}"


def render_all_members(container: object) -> str:
    docstring = inspect.getdoc(container)
    members = []
    keys_with_values = set(dir(container))
    keys_with_annotations = set(container.__annotations__.keys())
    for key in keys_with_values + keys_with_annotations:
        members.append(render_member(container, key, getattr(container, key, None)))
    return Template(
        dedent(
            '''\
            {% if docstring %}"""{{ docstring }}"""{% endif %}
            {% for member in members %}
            {{ member }}
            {% endfor %}
            '''
        )
    )


def render_module(m: ModuleType) -> str:
    return render_block(head=f"module {m.__name__}", body=render_all_members(m))


def render_class(c: type) -> str:
    return render_block(head=f"class {c.__name__}", body=render_all_members(c))


def render_stacktrace(
    skip=0,
    depth=None,
    include_private_symbols=False,
    symbol_trunc=50,
    use_qualname=True,
) -> str:
    formatted_trace = []

    stack = inspect.stack()

    if len(stack) <= skip:
        raise ValueError("`skip` must be less than the stack depth")

    for frame_record in stack[skip:]:
        frame, filename, line_no, function, lines, index = frame_record
        if depth is not None and len(formatted_trace) >= depth:
            break

        fn_name = function.__qualname__ if use_qualname else function.__name__

        frame_info = inspect.getframeinfo(frame)
        frame_locals = frame.f_locals
        frame_globals = frame.f_globals

        if not include_private_symbols:
            frame_locals = {
                k: v for k, v in frame_locals.items() if not k.startswith("_")
            }
            frame_globals = {
                k: v for k, v in frame_globals.items() if not k.startswith("_")
            }
        if symbol_trunc is not None:
            frame_locals = {k[:symbol_trunc]: v for k, v in frame_locals.items()}
            frame_globals = {k[:symbol_trunc]: v for k, v in frame_globals.items()}

        # Format the function call with arguments and local variables
        arg_info = inspect.getargvalues(frame)
        args = arg_info.args
        kwargs = {k: frame.f_locals[k] for k in arg_info.locals if k in arg_info.args}
        formatted_call = render_invocation(fn_name, args=args, kwargs=kwargs)

        template = Template(
            dedent(
                """
                File "{{ filepath }}", line: {{ line_no }}, in {{ formatted_call }}
                > {{ line }}
                Locals: {{ frame_locals }}
                Globals: {{ frame_globals }}
                """
            )
        )
        rendered_stacktrace = template.render(
            filepath=filename,
            line_no=line_no,
            formatted_call=formatted_call,
            line=lines[index].rstrip(),  # keep tab indentation
            frame_locals=frame_locals,
            frame_globals=frame_globals,
        )
        formatted_trace.append(rendered_stacktrace)

    return formatted_trace
