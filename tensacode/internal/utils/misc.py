def advanced_equality_check(*objects):
    """
    Perform an advanced equality check on multiple objects.

    This function compares objects by their attributes, ignoring None values.
    All objects must have a '__fields__' attribute.

    None is a wildcard that matches any value. If one of the objects has a None value for an attribute,
    the equality check will ignore that attribute for that object, but still consider that attribute
    on the other objects.

    Args:
        *objects: Variable number of objects to compare.

    Returns:
        bool: True if all non-None attributes are equal across objects, False otherwise.

    Raises:
        AttributeError: If any object doesn't have '__fields__' attribute.

    Examples:
        >>> from pydantic import BaseModel
        >>> class TestModel(BaseModel):
        ...     a: int
        ...     b: str
        >>> obj1 = TestModel(a=1, b='test')
        >>> obj2 = TestModel(a=1, b='test')
        >>> obj3 = TestModel(a=1, b=None)
        >>> advanced_equality_check(obj1, obj2)
        True
        >>> advanced_equality_check(obj1, obj2, obj3)
        True
        >>> obj4 = TestModel(a=2, b='test')
        >>> advanced_equality_check(obj1, obj4)
        False
    """
    if not objects:
        return True

    first = objects[0]

    # Ensure all objects have the same attributes
    if not all(hasattr(obj, "__fields__") for obj in objects):
        raise AttributeError("All objects must have '__fields__' attribute")

    for attr in first.__fields__:
        values = [getattr(obj, attr) for obj in objects]
        non_none_values = [v for v in values if v is not None]

        # If all values are None, skip this attribute
        if not non_none_values:
            continue

        # If non-None values are not all equal, return False
        if not all(v == non_none_values[0] for v in non_none_values):
            return False

    # All attributes are equal (ignoring None values)
    return True
