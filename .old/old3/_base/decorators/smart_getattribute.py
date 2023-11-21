from functools import wraps


def smart_getattribute(obj_or_cls):
    old_getattribute = getattr(obj_or_cls, "__getattribute__", object.__getattribute__)
    # early exit if we're already smart
    if (
        hasattr(old_getattribute, "__smart_getattribute__")
        and old_getattribute.__smart_getattribute__ is True
    ):
        return obj_or_cls
    # make smart function
    @wraps(old_getattribute)
    def __getattribute__(self, name):
        try:
            # try to get the attribute
            return old_getattribute(self, name)
        except AttributeError:
            # if it's not there, try to get it using the text model
            pass

    setattr(__getattribute__, "__smart_getattribute__", True)
    setattr(obj_or_cls, "__getattribute__", __getattribute__)
    return obj_or_cls
