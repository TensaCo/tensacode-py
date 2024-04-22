from typing import List

NoneType = type(None)


def exactly_one(*args):
    return exactly_n(*args, n=1)


def exactly_n(*args: List[object | NoneType], n: int):
    return sum(int(bool(arg)) for arg in args) == n


def stack_fields(base, *items):
    """
    Merges the unset fields of the base object with the values from the provided items.

    This function iterates over each item in the provided items list. For each item, it checks the base object for any fields that are set to None. If the item has a non-None value for an unset field in the base object, that value is set on the base object for that field.

    Args:
        base: The base object whose unset fields are to be populated.
        items: A variable number of objects from which to source the values for the unset fields of the base object.

    Returns:
        The base object with its unset fields populated with values from the provided items, if available.
    """
    if not items:
        return base
    for item in items:
        unset_keys = [k for k in dir(base) if getattr(base, k) is None]
        for k in unset_keys:
            if hasattr(item, k):
                setattr(base, k, getattr(item, k))
    return base


def get_inheritance_chain(base_class, child_class):
    """
    gets all the classes in the inheritance chain between the base class and the child class including the base class and the child class
    """
    visited = set()
    inheritance_tree = []

    def dfs(current_class):
        if current_class in visited:
            return
        visited.add(current_class)

        if issubclass(current_class, base_class) and current_class != base_class:
            inheritance_tree.append(current_class)

        for parent_class in current_class.__bases__:
            dfs(parent_class)

    dfs(child_class)
    return inheritance_tree
