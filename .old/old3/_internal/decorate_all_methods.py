 from old3._external import inspect_mate


def decorate_all_members(decorator_fn, condition=None, exclude=None):
    """Decorator that decorates all methods of a class with a given decorator."""

    def decorator(cls):
        for name, member in inspect_mate._get_members(cls):
            if exclude is not None and name in exclude:
                continue
            if condition is None or condition(cls, name, member):
                setattr(cls, name, decorator_fn(member))
        return cls

    return decorator


def decorate_all_attributes(decorator_fn, exclude=None):
    """Decorator that decorates all attributes of a class with a given decorator."""
    return decorate_all_members(
        decorator_fn,
        condition=lambda cls, name, member: inspect_mate.is_attribute(member),
        exclude=exclude,
    )


def decorate_all_property_methods(decorator_fn, exclude=None):
    """Decorator that decorates all property methods of a class with a given decorator."""
    return decorate_all_members(
        decorator_fn,
        condition=lambda cls, name, member: inspect_mate.is_property_method(member),
        exclude=exclude,
    )


def decorate_all_regular_methods(decorator_fn, exclude=None):
    """Decorator that decorates all regular methods of a class with a given decorator."""
    return decorate_all_members(
        decorator_fn,
        condition=lambda cls, name, member: inspect_mate.is_regular_method(member),
        exclude=exclude,
    )


def decorate_all_static_methods(decorator_fn, exclude=None):
    """Decorator that decorates all static methods of a class with a given decorator."""
    return decorate_all_members(
        decorator_fn,
        condition=lambda cls, name, member: inspect_mate.is_static_method(member),
        exclude=exclude,
    )


def decorate_all_class_methods(decorator_fn, exclude=None):
    """Decorator that decorates all class methods of a class with a given decorator."""
    return decorate_all_members(
        decorator_fn,
        condition=lambda cls, name, member: inspect_mate.is_class_method(member),
        exclude=exclude,
    )
