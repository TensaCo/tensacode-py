from enum import Enum, auto
import types
from typing import Any, Type, overload
import typing

from old3._external import inspect_mate


class PyObjType(Enum):
    module = auto()
    klass = auto()
    lmbda = auto()
    staticmethod = auto()
    classmethod = auto()
    property = auto()
    method = auto()
    function = auto()
    attribute = auto()
    object = auto()
    unknown = auto()

    @staticmethod
    def of(pyobj):
        if isinstance(pyobj, types.ModuleType):
            return PyObjType.module
        if isinstance(pyobj, typing.Type):
            return PyObjType.klass
        if isinstance(pyobj, types.LambdaType):
            return PyObjType.lmbda
        if not hasattr(pyobj, "__name__"):
            # just making do with the information we have
            if isinstance(pyobj, types.MethodType):
                return PyObjType.method
            if isinstance(pyobj, types.FunctionType):
                return PyObjType.function
            if isinstance(pyobj, object):
                return PyObjType.object
            return PyObjType.unknown
        try:
            klass = pyobj if isinstance(pyobj, Type) else pyobj.__class__
        except:
            pass
        if inspect_mate.is_static_method(klass, pyobj.__name__, pyobj):
            return PyObjType.staticmethod
        if inspect_mate.is_class_method(klass, pyobj.__name__, pyobj):
            return PyObjType.classmethod
        if inspect_mate.is_property_method(klass, pyobj.__name__, pyobj):
            return PyObjType.property
        if inspect_mate.is_regular_method(klass, pyobj.__name__, pyobj):
            return PyObjType.method
        if isinstance(pyobj, types.FunctionType | types.CoroutineType):
            return PyObjType.function
        if inspect_mate.is_attribute(klass, pyobj.__name__, pyobj):
            return PyObjType.attribute
        if isinstance(pyobj, object):
            return PyObjType.object
        return PyObjType.unknown

    @property
    def type(self):
        if self == PyObjType.module:
            return types.ModuleType
        if self == PyObjType.klass:
            return typing.Type
        if self == PyObjType.lmbda:
            return types.LambdaType
        if self == PyObjType.staticmethod:
            return types.FunctionType
        if self == PyObjType.classmethod:
            return types.FunctionType
        if self == PyObjType.property:
            return property
        if self == PyObjType.method:
            return types.MethodType
        if self == PyObjType.function:
            return types.FunctionType
        if self == PyObjType.attribute:
            return object
        if self == PyObjType.object:
            return object
        if self == PyObjType.unknown:
            return typing.Any
        raise ValueError(f"Unknown PyObjType: {self}")


def encode(obj: Any, depth=1, prompt_extra=None, include_hidden=False):
    """encode a python object into a string

    Args:
        obj (Any): The object to encode.
        depth (int, optional): How much to recursively encode. 0 means just provide a summary of the object. Use `inf` for infinite, although this might not halt. Defaults to 1.
        prompt_extra (Any, optional): Anything other information you think would be useful for conditioning the encoding. Defaults to None.
        include_hidden (bool, optional): Whether to include / recursively expand hidden attributes/functions/classes/modules (ones that start with '_'). Defaults to False.

    Returns:
        str: The encoded object.
    """
    pass


@overload
def encode_class(
    obj: PyObjType.klass.type, depth=1, prompt_extra=None, include_hidden=False
):
    subclasses = ", ".join(sub.__name__ for sub in obj.__subclasses__())
    contents = list(filter(lambda f: not f.startswith("_") or include_hidden, dir(obj)))
    return f"class {obj.__name__}({subclasses}):\n  " + "\n  ".join(
        text.encode(field) for field in fields
    )


@encode.add
def encode_module(
    obj: PyObjType.module.type, depth=1, prompt_extra=None, include_hidden=False
):
    contents = []
    for name in dir(obj):
        if name.startswith("_") and not include_hidden:
            continue
        attr = getattr(obj, name)
        contents.append(f"{name} = {text.encode(attr)}")
    return "\n".join(contents)


def get_contents(module, include_hidden):
    yield from filter(lambda f: not f.startswith("_") or include_hidden, dir(module))


@encode.add
def encode_function(
    obj: PyObjType.function.type, depth=1, prompt_extra=None, include_hidden=False
):
    return f"def {obj.__name__}({inspect.signature(obj)}):\n  {text.encode(obj.__code__)}"


@encode.add
def encode_staticmethod(
    obj: PyObjType.staticmethod.type, depth=1, prompt_extra=None, include_hidden=False
):
    return text.encode(obj.__func__)


@encode.add
def encode_classmethod(
    obj: PyObjType.classmethod.type, depth=1, prompt_extra=None, include_hidden=False
):
    return f"@classmethod\ndef {obj.__name__}(cls, {inspect.signature(obj)}):\n  {text.encode(obj.__func__.__code__)}"


@encode.add
def encode_property(
    obj: PyObjType.property.type, depth=1, prompt_extra=None, include_hidden=False
):
    return (
        f"@property\ndef {obj.fget.__name__}(self):\n  {text.encode(obj.fget.__code__)}"
    )


@encode.add
def encode_method(
    obj: PyObjType.method.type, depth=1, prompt_extra=None, include_hidden=False
):
    return f"def {obj.__name__}(self, {inspect.signature(obj)}):\n  {text.encode(obj.__code__)}"


@encode.add
def encode_attribute(
    obj: PyObjType.attribute.type, prompt_extra=None, include_hidden=False
):
    return repr(obj)


@encode.add
def encode_object(
    obj: PyObjType.object.type, depth=1, prompt_extra=None, include_hidden=False
):
    if depth == 0:
        return str(obj)


@encode.add
def encode_unknown(
    obj: PyObjType.unknown.type, depth=1, prompt_extra=None, include_hidden=False
):
    return repr(obj)


@encode.add
def encode_none(obj: None, depth=1, prompt_extra=None, include_hidden=False):
    return "None"
