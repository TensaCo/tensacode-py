"""JSON-safe configuration identities, deliberately excluding learned values."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import nn

from .latent import Space


def qualified_name(value: Any) -> str:
    target = value if isinstance(value, type) else type(value)
    return f"{target.__module__}.{target.__qualname__}"


def _json_value(value: Any, *, path: str) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_value(item, path=f"{path}[]") for item in value]
    if isinstance(value, Mapping) and all(isinstance(key, str) for key in value):
        return {
            key: _json_value(item, path=f"{path}.{key}")
            for key, item in sorted(value.items())
        }
    raise ValueError(
        f"{path} is not JSON-safe configuration metadata; provide an explicit "
        "configuration() that excludes learned tensor values"
    )


def _explicit_configuration(value: Any) -> Any | None:
    configuration = getattr(value, "configuration", None)
    if callable(configuration):
        return _json_value(configuration(), path=f"{qualified_name(value)}.configuration")
    return None


def callable_identity(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    explicit = _explicit_configuration(value)
    if explicit is not None:
        return {"callable": qualified_name(value), "configuration": explicit}
    if hasattr(value, "__module__") and hasattr(value, "__qualname__"):
        if (
            getattr(value, "__closure__", None)
            or "<locals>" in value.__qualname__
            or value.__qualname__ == "<lambda>"
            or getattr(value, "__self__", None) is not None
        ):
            raise ValueError(
                "Closure, local, lambda, and bound callbacks require explicit "
                "configuration() metadata for persistence"
            )
        result = {"callable": f"{value.__module__}.{value.__qualname__}"}
        if hasattr(value, "__defaults__"):
            result["defaults"] = _json_value(
                value.__defaults__ or (), path=f"{result['callable']}.__defaults__"
            )
            result["keyword_defaults"] = _json_value(
                value.__kwdefaults__ or {}, path=f"{result['callable']}.__kwdefaults__"
            )
        return result
    raise ValueError(
        "Stateful callable objects require explicit configuration() metadata for persistence"
    )


def space_configuration(space: Space | None) -> dict[str, Any] | None:
    return None if space is None else space.configuration()


def module_configuration(module: nn.Module) -> dict[str, Any]:
    """Describe module architecture and tensor schemas, never tensor values."""

    framework_fields = {
        "training",
        "_parameters",
        "_buffers",
        "_non_persistent_buffers_set",
        "_backward_pre_hooks",
        "_backward_hooks",
        "_is_full_backward_hook",
        "_forward_hooks",
        "_forward_hooks_with_kwargs",
        "_forward_hooks_always_called",
        "_forward_pre_hooks",
        "_forward_pre_hooks_with_kwargs",
        "_state_dict_hooks",
        "_state_dict_pre_hooks",
        "_load_state_dict_pre_hooks",
        "_load_state_dict_post_hooks",
        "_modules",
    }

    def attributes(child: nn.Module) -> dict[str, Any]:
        result = {}
        for name, value in sorted(vars(child).items()):
            if name in framework_fields:
                continue
            result[name] = _json_value(
                value,
                path=f"{qualified_name(child)}.{name}",
            )
        return result

    modules = []
    for path, child in module.named_modules():
        explicit = _explicit_configuration(child)
        entry = {
            "path": path,
            "type": qualified_name(child),
            "configuration": explicit,
        }
        if explicit is None:
            entry["attributes"] = attributes(child)
        modules.append(entry)

    return {
        "type": qualified_name(module),
        "configuration": _explicit_configuration(module),
        "modules": modules,
        "parameters": [
            {
                "name": name,
                "shape": list(parameter.shape),
                "dtype": str(parameter.dtype),
                "requires_grad": parameter.requires_grad,
            }
            for name, parameter in module.named_parameters()
        ],
        "buffers": [
            {"name": name, "shape": list(buffer.shape), "dtype": str(buffer.dtype)}
            for name, buffer in module.named_buffers()
        ],
    }
