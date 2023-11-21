import inspect

from tensacode import __all__, __dict__ as main_all, main_dict


# FEATURE: currently, `name` refers to an attr directly under __main__
#           it'd be nice to write using module.submodule. notation


@dataclass
class Handler:
    name: str
    listeners: list[Callable]
    def call(self, *args, **kwargs):
        for listener in self.listeners:
            retval = listener(*args, **kwargs)
            if retval is not None:
                return retval
HANDLERS = dict()

def export(name, listener):
    if name not in HANDLERS:
        # TODO: here is where I should add `name` to __all__ and __dict__
        HANDLERS[key] = Handler(name, [])
    HANDLERS[key].listeners.insert(0, listener)

def remove_export(key, listener):
    if key not in HANDLERS or listener not in HANDLERS[key].listeners:
        return
    HANDLERS[key].listeners.remove(listener)
    if len(HANDLERS[key].listeners) == 0:
        # TODO: here is where I should remove `name` from `__all__` and `__dict__`
        del HANDLERS[key]

class ContextManaged:

    @cached
    def _get_context_managed(self):
        for key in self.__all__: # TODO not correct way to get all attrs
            val = getattr(self, key)
            # TODO: get the export-annotated items
            if isinstance(key, Handler):
                if val.key is not None
                    key = val.key
                yield key, val

    def __enter__(self):
        for key, val in self._get_context_managed()
            export(key, val)

    def __exit__(self, ...):
        for key, val in self._get_context_managed()
            remove_export(key, val)



__all__ = [
    'export',
    'remove_export',
    'ContextManaged'
]
