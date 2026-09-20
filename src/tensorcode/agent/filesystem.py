"""Observable, root-confined filesystem actions for explicit task specifications.

No project convention or language interpretation lives here. Grounding supplies
ancestors from path structure; the planner chooses and orders declared actions.
Writes exclusively create files. Replacing existing content needs a separately
authorized action model, which this plugin deliberately does not provide.
"""
from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping

from ..goals import Condition, GoalSpec
from ..derivations import validate_record_support, validate_record_supports
from ..records import Evidence, Interval, Proposition, Ref, Store
from ..outcomes import Receipt, Unknown
from .plugin import Call, Capability, Effect, Param, Plugin, Precondition

if TYPE_CHECKING:
    from .refinements import RefinementLibrary


@dataclass(frozen=True)
class _ResourceAncestor:
    """Mechanical ancestor retaining the supplied resource's authorization.

    This internal plan value is distinct from user relative paths: those never
    admit traversal. Resolution still confines every ancestor to the adapter root.
    """

    resource: Ref
    levels: int


class FileSystemPlugin(Plugin):
    """Operate beneath an existing directory, using exact UTF-8 text content.

    Use consistent path representations in a specification: root-relative strings
    are convenient. Path objects and absolute paths are accepted too. Symbolic
    links are unsupported, including links whose targets remain inside the root.
    This is a local filesystem adapter, not an isolation boundary against another
    process concurrently replacing directories during an operation.
    """

    def __init__(self, root: Path, *, name: str = "filesystem",
                 refinements: RefinementLibrary | None = None) -> None:
        super().__init__(name=name, planning_enabled=True)
        from .refinements import RefinementLibrary

        if refinements is not None and not isinstance(refinements, RefinementLibrary):
            raise TypeError("refinements must be a RefinementLibrary or None")
        self.refinements = refinements
        self.resources = Store()
        self._resource_paths: dict[Ref, Path] = {}
        given = Path(root).absolute()
        if any(part.is_symlink() for part in (given, *given.parents)):
            raise ValueError("filesystem root must not traverse symbolic links")
        self.root = given.resolve(strict=True)
        if not self.root.is_dir():
            raise ValueError("filesystem root must be a directory")

    def refine_goal(self, goal: Any) -> GoalSpec | Unknown:
        """Apply only the explicitly supplied domain library, if any.

        No domain recipes are bundled or loaded by this adapter.
        The filesystem action and observation model is independent of recipes.
        """
        if self.refinements is None:
            return Unknown("no_refinement", "filesystem refinements are disabled")
        return self.refinements.refine(goal, context={"root": self.root})

    def capabilities(self) -> tuple[Capability, ...]:
        return (
            Capability("mkdir", (Param("path", "path"), Param("parent", "path")),
                       effects=(Effect("directory_exists", {"path": "path"}),
                                Effect("path_exists", {"path": "path"})),
                       preconditions=(Precondition("directory_exists", {"path": "parent"}),
                                      Precondition("path_exists", {"path": "path"}, negated=True))),
            Capability("write_file", (Param("path", "path"), Param("parent", "path"), Param("text", "text")),
                       effects=(Effect("file_exists", {"path": "path"}),
                                Effect("path_exists", {"path": "path"}),
                                Effect("content", {"path": "path", "text": "text"})),
                       preconditions=(Precondition("directory_exists", {"path": "parent"}),
                                      Precondition("path_exists", {"path": "path"}, negated=True))),
        )

    def bind_resource(self, resource: Ref, path: str, *, binding: Ref,
                      evidence: Evidence) -> Proposition:
        """Retain supplied realization; a Ref cannot change paths in this lifetime.

        Withdrawal disables use, including dispatch of already retained calls.
        This is explicit knowledge, not grounding inferred from reference text.
        """
        if not isinstance(resource, Ref) or not isinstance(binding, Ref):
            raise ValueError("resource and binding must be references")
        if not isinstance(path, str) or not isinstance(evidence, Evidence):
            raise ValueError("resource binding needs a literal path and explicit evidence")
        if not isinstance(evidence.source, Ref) or evidence.derived_from:
            raise ValueError("resource binding requires direct supplied evidence")
        resolved = self._path(path)
        previous = self._resource_paths.get(resource)
        if previous is not None and previous != resolved:
            raise ValueError("resource cannot be rebound to another path")
        proposition = Proposition("filesystem_resource_binding", {
            "resource": resource, "path": str(resolved), "binding": binding,
        })
        self.resources.assert_(proposition, evidence)
        self._resource_paths[resource] = resolved
        return proposition

    def _resource_path(self, resource: Ref) -> Path:
        records = [record for record in self.resources.propositions("filesystem_resource_binding")
                   if record.proposition.roles.get("resource") == resource]
        if len(records) != 1:
            raise ValueError("resource needs one active unambiguous binding")
        record = records[0]
        proposition = record.proposition
        if (set(proposition.roles) != {"resource", "path", "binding"}
                or not isinstance(proposition.roles["binding"], Ref)
                or not isinstance(proposition.roles["path"], str)
                or proposition.scope is not None or proposition.valid != Interval()
                or proposition.polarity is not True or proposition.modality != "asserted"):
            raise ValueError("unsupported resource binding qualifiers or roles")
        if validate_record_support(self.resources, record.id) is not True:
            raise ValueError("resource binding support is unavailable")
        path = self._path(proposition.roles["path"])
        if self._resource_paths.get(resource) != path:
            raise ValueError("resource binding does not match its supplied lifetime path")
        return path

    def _path(self, value: Any) -> Path:
        if isinstance(value, _ResourceAncestor):
            if type(value.levels) is not int or value.levels < 1:
                raise ValueError("resource ancestor requires a positive depth")
            base = self._resource_path(value.resource)
            if value.levels > len(base.relative_to(self.root).parts):
                raise ValueError("resource ancestor exceeds the filesystem root")
            return self._path(base.parents[value.levels - 1])
        if isinstance(value, Ref):
            return self._resource_path(value)
        if isinstance(value, dict):
            if (set(value) != {"root", "relative"} or not isinstance(value["root"], Ref)
                    or not isinstance(value["relative"], str)):
                raise ValueError("structural path requires a root reference and relative string")
            relative = Path(value["relative"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("structural path must remain beneath its resource")
            return self._path(self._resource_path(value["root"]) / relative)
        if not isinstance(value, (str, Path)):
            raise ValueError("path must be a string or Path")
        raw = Path(value)
        if ".." in raw.parts:
            raise ValueError("parent traversal is unsupported")
        path = raw if raw.is_absolute() else self.root / raw
        path.relative_to(self.root)
        if any(part.is_symlink() for part in (path, *path.parents)):
            raise ValueError("symbolic links are unsupported")
        return path

    def _parent(self, value: Any) -> Any:
        if isinstance(value, _ResourceAncestor):
            self._path(value)
            return _ResourceAncestor(value.resource, value.levels + 1)
        if isinstance(value, Ref):
            self._resource_path(value)
            return _ResourceAncestor(value, 1)
        if isinstance(value, dict):
            self._path(value)
            relative = Path(value["relative"])
            if relative == Path("."):
                return _ResourceAncestor(value["root"], 1)
            parent = relative.parent
            return value["root"] if parent == Path(".") else {"root": value["root"], "relative": str(parent)}
        parent = Path(value).parent
        return parent if isinstance(value, Path) else str(parent)

    def enumerate_actions(self, goal: GoalSpec) -> Iterable[Call]:
        directories: list[Any] = []
        writes: list[tuple[Any, str]] = []
        contents = [c for c in goal.conditions if c.pred == "content" and not c.negated]
        for condition in goal.conditions:
            if condition.negated or condition.pred not in {"directory_exists", "file_exists", "content"}:
                continue
            value = condition.args.get("path")
            try:
                self._path(value)
            except (ValueError, OSError):
                continue
            directory = value if condition.pred == "directory_exists" else self._parent(value)
            while self._path(directory) != self.root:
                if directory not in directories:
                    directories.append(directory)
                directory = self._parent(directory)
            if condition.pred == "content" and isinstance(condition.args.get("text"), str):
                writes.append((value, condition.args["text"]))
            elif condition.pred == "file_exists" and not any(c.args.get("path") == value for c in contents):
                writes.append((value, ""))
        for directory in reversed(directories):
            yield Call(self.name, "mkdir", (("path", directory), ("parent", self._parent(directory))))
        seen: list[tuple[Any, str]] = []
        for path, content in writes:
            if (path, content) not in seen:
                seen.append((path, content))
                yield Call(self.name, "write_file", (("path", path), ("parent", self._parent(path)), ("text", content)))

    def observe_condition(self, condition: Condition) -> bool | Unknown:
        expected = {"directory_exists": {"path"}, "file_exists": {"path"},
                    "path_exists": {"path"}, "content": {"path", "text"}}
        if condition.pred not in expected or set(condition.args) != expected[condition.pred]:
            return Unknown("unsupported_condition", condition.pred)
        try:
            path = self._path(condition.args["path"])
            if condition.pred == "path_exists":
                holds = path.exists()
            elif condition.pred == "directory_exists":
                holds = path.is_dir()
            elif condition.pred == "file_exists":
                holds = path.is_file()
            else:
                text = condition.args["text"]
                if not isinstance(text, str):
                    return Unknown("invalid_content", "content must be text")
                holds = path.is_file() and path.read_bytes() == text.encode("utf-8")
        except (OSError, ValueError) as exc:
            return Unknown("unobserved", str(exc))
        return not holds if condition.negated else holds

    def precondition_holds(self, condition: Precondition, args: Mapping[str, Any]) -> bool | Unknown:
        if any(param not in args for param in condition.roles.values()):
            return Unknown("unbound_precondition", condition.pred)
        return self.observe_condition(Condition(condition.pred, {role: args[param] for role, param in condition.roles.items()}, condition.negated))

    def holds(self, cap: Capability, args: Mapping[str, Any]) -> bool | Unknown:
        unknown = None
        for effect in cap.effects:
            result = self.precondition_holds(effect, args)
            if result is False:
                return False
            if isinstance(result, Unknown):
                unknown = result
        return unknown if unknown is not None else True

    def execute(self, act: Call, *, key: str | None) -> Receipt:
        cap = next((cap for cap in self.capabilities() if cap.name == act.capability), None)
        args = dict(act.args)
        if act.plugin != self.name or cap is None or set(args) != {p.name for p in cap.params}:
            return Receipt(act, "rejected", error="unknown or malformed filesystem action")
        try:
            path = self._path(args["path"])
            parent = self._path(args["parent"])
            if path.parent != parent:
                raise ValueError("parent binding does not match path")
            for condition in cap.preconditions:
                if self.precondition_holds(condition, args) is not True:
                    raise ValueError(f"precondition {condition.pred} is not established")
            if cap.name == "write_file":
                if not isinstance(args["text"], str):
                    raise ValueError("content must be text")
                data = args["text"].encode("utf-8")
            # Preconditions can invoke supplied observation hooks. Do not carry
            # cached realization across those callbacks into a filesystem write.
            bindings = deepcopy(self.resources.propositions())
            if self._path(args["path"]) != path or self._path(args["parent"]) != parent:
                raise ValueError("filesystem realization changed during preconditions")
            resources = set()
            for value in (args["path"], args["parent"]):
                if isinstance(value, Ref):
                    resources.add(value)
                elif isinstance(value, _ResourceAncestor):
                    resources.add(value.resource)
                elif isinstance(value, dict):
                    resources.add(value["root"])
            supports = tuple(record.id for record in bindings
                             if record.proposition.predicate == "filesystem_resource_binding"
                             and record.proposition.roles.get("resource") in resources)
            if supports and validate_record_supports(self.resources, supports) is not True:
                raise ValueError("resource support is unavailable at dispatch")
            # Validating the parent can invalidate the already checked target.
            # Store.revision does not track assert_/supersede, so compare records.
            current_bindings = [record for _, record in sorted(self.resources._props.items())
                                if record.retracted is None]
            if bindings != current_bindings:
                raise ValueError("resource binding support changed during dispatch validation")
        except (ValueError, OSError) as exc:
            return Receipt(act, "rejected", idempotency_key=key, error=str(exc))
        try:
            if cap.name == "mkdir":
                path.mkdir()
            else:
                # O_EXCL prevents accidental replacement even if another writer
                # creates the file after the precondition observation.
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
                with os.fdopen(fd, "wb") as stream:
                    stream.write(data)
            return Receipt(act, "applied", idempotency_key=key)
        except OSError as exc:
            return Receipt(act, "failed", idempotency_key=key, error=str(exc))
