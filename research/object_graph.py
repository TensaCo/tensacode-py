"""Object-graph import/export on top of the record codec. Research code, not part of the core.

Kept to reproduce the legacy-TCIR comparison (docs/revival/02-representation.md).
Deferred from the package: no current program needs to lift arbitrary Python object
graphs into records.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Iterable, Iterator

from tensorcode.records import ConversionReport, Ref, TypeRegistry, _fields_of, _is_model, decode, encode



@dataclass(frozen=True)
class Conversion:
    records: dict[Ref, Any]  # encoded payloads, entity-to-entity links as {"$ref": ...}
    roots: tuple[Ref, ...]
    report: ConversionReport


def to_records(roots: Iterable[Any], registry: TypeRegistry) -> Conversion:
    """Split an object graph into one record per entity instance.

    Identity follows Python object identity *and* the registered identity key:
    the same object reached twice is one record; two distinct objects that claim
    the same identity key are reported, not silently merged.
    """
    refs: dict[int, Ref] = {}
    owners: dict[Ref, int] = {}
    seen: dict[int, str] = {}
    pending: list[Any] = []
    report = ConversionReport()

    def ref_for(obj: Any) -> Ref:
        if id(obj) in refs:
            return refs[id(obj)]
        reg = registry.lookup_cls(type(obj))
        assert reg and reg.identity
        key = reg.identity(obj)
        ref = Ref(f"{reg.name}:{key}")
        if ref in owners and owners[ref] != id(obj):
            report.losses.append(f"identity collision: two distinct {reg.name} objects share key {key!r}")
            ref = Ref(f"{reg.name}:{key}#{len(owners)}")
        refs[id(obj)], owners[ref] = ref, id(obj)
        pending.append(obj)
        return ref

    root_refs = tuple(ref_for(r) for r in roots)
    records: dict[Ref, Any] = {}
    while pending:
        obj = pending.pop()
        ref = refs[id(obj)]
        if ref in records:
            continue
        enc = encode(obj, registry, entity_ref=ref_for, root_path=str(ref), seen=seen)
        report.losses.extend(enc.report.losses)
        report.opaque.extend(f"{ref}: {o}" for o in enc.report.opaque)
        records[ref] = enc.data
    return Conversion(records, root_refs, report)


def from_records(conversion: Conversion, registry: TypeRegistry) -> tuple[list[Any], ConversionReport]:
    """Rebuild objects, restoring shared identity and entity cycles.

    Two passes: construct every entity with placeholder links, then patch links.
    Frozen dataclasses and validated models cannot be patched after construction;
    cycles through them are reported, not faked.
    """
    report = ConversionReport()
    built: dict[Ref, Any] = {}
    links: list[tuple[Any, list[str | int], Ref]] = []

    def placeholder(ref: Ref) -> Ref:
        return ref

    for ref, data in conversion.records.items():
        obj, rep = decode(data, registry, resolve=placeholder)
        report.opaque.extend(f"{ref}: {o}" for o in rep.opaque)
        built[ref] = obj

    def walk(container: Any, path: list[str | int]) -> Iterator[tuple[list[str | int], Ref]]:
        if isinstance(container, Ref):
            yield path, container
        elif isinstance(container, list):
            for i, x in enumerate(container):
                yield from walk(x, path + [i])
        elif isinstance(container, dict):
            for k, x in container.items():
                yield from walk(x, path + [k])
        elif dataclasses.is_dataclass(container) or _is_model(container):
            for k, x in _fields_of(container).items():
                yield from walk(x, path + [k])

    for obj in built.values():
        for path, target in walk(obj, []):
            if target in built:
                links.append((obj, path, target))

    for obj, path, target in links:
        parent = obj
        for step in path[:-1]:
            parent = parent[step] if isinstance(parent, (list, dict)) else getattr(parent, step)
        last = path[-1]
        try:
            if isinstance(parent, (list, dict)):
                parent[last] = built[target]
            elif dataclasses.is_dataclass(parent) and type(parent).__dataclass_params__.frozen:
                report.losses.append(f"cannot link frozen {type(parent).__name__}.{last} -> {target}; left as Ref")
            else:
                setattr(parent, last, built[target])
        except (AttributeError, TypeError, ValueError) as exc:
            report.losses.append(f"cannot link {type(parent).__name__}.{last} -> {target}: {exc}")
    return [built[r] for r in conversion.roots], report


