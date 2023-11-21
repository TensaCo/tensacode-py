from old3.utils.pyobj import PyObjType


@overload
def encode_class(obj: PyObjType.klass.type, include_hidden=False):
    subclasses = ", ".join(sub.__name__ for sub in obj.__subclasses__())
    fields = []
    for attr_name in dir(obj):
        if not include_hidden and attr_name.startswith("_"):
            continue
        attr = getattr(obj, attr_name)
        if isinstance(attr, property):
            attrs.append(f"{attr_name}: {attr.fget.__annotations__['return'].__name__}")
        else:
            attrs.append(attr_name)
    print(f"class {obj.__name__}({subclasses}):\n  " + "\n  ".join(attrs))
