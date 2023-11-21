import sys
import importlib
import inspect


def import_all_from_public_submodules_recursively():
    caller_frame = inspect.stack()[1]
    caller_module = inspect.getmodule(caller_frame[0])
    _recursive_import(caller_module, [caller_module.__dict__])


def _recursive_import(module, namespaces):
    """DF import all public submodules and their public members"""
    public_submodules = []

    for name, obj in inspect.getmembers(module):
        if inspect.ismodule(obj) and not name.startswith("_"):
            public_submodules.append(obj)

    for submodule in public_submodules:
        for name, obj in inspect.getmembers(submodule):
            if not name.startswith("_"):
                for namespace in namespaces:
                    namespace[name] = obj

        # Recursively import from submodules
        _recursive_import(submodule, namespaces + [submodule.__dict__])
