def add_extras(dict):
    """Add extra members to a class.

    Example:
    >>> @add_extras({"foo": "bar"})
        class Foo:
            pass
    >>> Foo.foo
    ... "bar"
    """

    def wrapper(cls):
        cls.__dict__.update(dict)
        return cls

    return wrapper
