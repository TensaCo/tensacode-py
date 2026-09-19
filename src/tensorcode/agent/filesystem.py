"""Observable, root-confined filesystem actions for explicit task specifications.

No project convention or language interpretation lives here. Grounding supplies
ancestors from path structure; the planner chooses and orders declared actions.
Writes exclusively create files. Replacing existing content needs a separately
authorized action model, which this plugin deliberately does not provide.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping

from ..goals import Condition, GoalSpec
from ..outcomes import Receipt, Unknown
from .plugin import Call, Capability, Effect, Param, Plugin, Precondition

if TYPE_CHECKING:
    from .refinements import RefinementLibrary


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

    def _path(self, value: Any) -> Path:
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

    @staticmethod
    def _parent(value: str | Path) -> str | Path:
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
