from typing import Union


def join(vals: list | tuple | dict, delim: str = ", ") -> str:
    if isinstance(vals, (list, tuple)):
        return delim.join(str(val) for val in vals)
    elif isinstance(vals, dict):
        return delim.join([f"{key}={val}" for key, val in vals.items()])
    else:
        raise TypeError()
