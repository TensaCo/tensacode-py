"""A desktop plugin on the computerworld engine: files, folders and apps, worked through the screen.

Everything this plugin does, it does as a user would: through the actor session's
terminal (launch it, click the shell input, confirm focus, type, press Enter, read the
output back off the engine's semantic channel). It never touches the privileged
filesystem API — that channel belongs to setup and grading.

What it tells the agent is data:

* **vocabulary** — the words a desktop needs that WordNet lacks or misreads ("folder"
  as a directory, the names of the installed apps);
* **kinds** — ``folder`` is a ``directory``; a ``directory`` and a ``file`` are both
  ``path``s; an app is an ``application``;
* **capabilities** — each with its effects in VerbNet's predicates over role groups.

How a description becomes a path is this plugin's own knowledge of its machine: which
directories exist under home (read at start, not hard-coded), and which files match a
name. Nothing here sees the user's words.
"""

from __future__ import annotations

import shlex
from typing import Any, Iterable, Mapping

from examples.browser_agents.perception.computerworld import CwProvider
from examples.browser_agents.perception.cw_body import CwBody
from tensorcode.agent.plugin import Call, Capability, Effect, Informs, Param, Plugin
from tensorcode.language import Entity, words
from tensorcode.outcomes import Receipt, Unknown
from tensorcode.records import Claim, Ref

HOME = "/home/agent"


def path_ref(path: str) -> Ref:
    return Ref(f"path:{path}")


def path_of(ref: Any) -> str | None:
    if isinstance(ref, Ref) and ref.id.startswith("path:"):
        return ref.id[len("path:"):]
    return None


CAPABILITIES = (
    Capability("make_directory", (Param("path", "directory"),),
               effects=(Effect("be", {"undergoer": "path"}),), description="mkdir -p"),
    Capability("make_file", (Param("path", "file"),),
               effects=(Effect("be", {"undergoer": "path"}),), description="touch"),
    Capability("delete", (Param("path", "path"),),
               effects=(Effect("has_location", {"undergoer": "path"}, negated=True),
                        Effect("destroyed", {"undergoer": "path"})), description="rm -r"),
    Capability("move", (Param("path", "path"), Param("destination", "directory")),
               effects=(Effect("has_location", {"undergoer": "path", "goal": "destination"}),
                        Effect("has_location", {"undergoer": "path"}, negated=True)), description="mv"),
    Capability("list_directory", (Param("directory", "directory"),),
               informs=(Informs("has_location", "goal", "directory"),), effect_kind="read", description="ls"),
    Capability("read_file", (Param("path", "file"),),
               informs=(Informs("contain", "undergoer", "path"),), effect_kind="read", description="cat"),
    Capability("open_application", (Param("app", "application"),),
               effects=(Effect("has_state", {"undergoer": "app"}),), description="click its launcher button"),
)


class DesktopPlugin(Plugin):
    """Owns one actor session on a computerworld machine. Use from the thread that made it."""

    def __init__(self, world, *, on_step=None) -> None:
        super().__init__(
            name="desktop",
            lexicon=tuple(words("folder", "directory", cat="N", sem="folder"))
            + tuple(words("file", cat="N", sem="file")),
            kinds={"folder": ("directory",), "directory": ("path",), "file": ("path",),
                   "document": ("file",), "note": ("file",), "report": ("file",)},
        )
        self.world = world
        self.surface = world.actor()
        self.body = CwBody(self.surface, CwProvider(), episode="desktop-plugin", on_step=on_step)
        self.places: dict[str, str] = {}
        self.apps: dict[str, Any] = {}
        self.log: list[tuple[str, list[str]]] = []
        self._look_around()

    # ------------------------------------------------------------------ the terminal

    def _shell(self):
        if not self.surface.has_terminal():
            self.surface.act("application.v1", "launch", {"kind": "terminal"})
        screen = self.body.observe()
        box = [c for c in screen.controls if c.role == "textbox" and c.name == "Shell input"]
        return box[0] if box else None

    def run(self, command: str) -> tuple[bool, list[str]]:
        """Type a command into the terminal and read what it printed."""
        shell = self._shell()
        if shell is None:
            return False, ["no terminal"]
        before = len(self.surface.terminal_lines())
        receipt = self.body.fill(shell, command, submit=True)
        out = self.surface.terminal_lines()[before:]
        self.log.append((command, out))
        return receipt.status == "applied", out

    def _look_around(self) -> None:
        """Learn this machine from looking at it: its apps (from the launcher), then its places."""
        screen = self.body.observe()
        # launcher buttons belong to no window; the top bar's are menus ("Open ...", "Show ...")
        launchers = [c for c in screen.controls if c.role == "button" and c.name and not c.section
                     and not c.name.lower().startswith(("open ", "show "))]
        self.apps = {c.name.lower(): c for c in launchers}
        self.kinds = {**self.kinds, **{name: ("application",) for name in self.apps}}
        self.lexicon = self.lexicon + tuple(e for name in self.apps for e in words(name.split()[-1], cat="N", sem=name.split()[-1]))
        self.places = {"home": HOME}
        for name, is_dir in self.listing(HOME):
            if is_dir:
                self.places[name.lower()] = f"{HOME}/{name}"
                self.kinds = {**self.kinds, name.lower(): ("directory",)}

    def listing(self, directory: str) -> list[tuple[str, bool]]:
        """A directory's entries, each checked for being a directory (this ls does not mark them)."""
        ok, out = self.run(f"ls -1A {shlex.quote(directory)}")
        if not ok:
            return []
        entries = []
        for name in out:
            if ":" in name and name.split(":", 1)[0] == "ls":
                return []
            _, test = self.run(f"test -d {shlex.quote(directory + '/' + name)} && echo d || echo f")
            entries.append((name, bool(test) and test[-1].strip() == "d"))
        return entries

    # ------------------------------------------------------------------ plugin protocol

    def capabilities(self):
        return CAPABILITIES

    def perceive(self) -> Iterable[Claim]:
        for name, path in self.places.items():
            if name not in ("home", "~"):
                yield Claim(path_ref(path), "is_a", "directory")

    def display(self, ref: Any) -> str:
        path = path_of(ref)
        return path.rsplit("/", 1)[-1] if path else None

    def denote(self, description: Any) -> Ref | Unknown:
        path = self._resolve(description, creating=False)
        return path_ref(path) if isinstance(path, str) else path

    def refer(self, description: Any, param: Param, *, context: Mapping[str, Any]) -> Ref | Unknown:
        if param.kind == "application":
            app = self.app_for(description)
            return Ref(f"app:{app}") if isinstance(app, str) else app
        creating = param.kind in ("directory", "file") and _is_new(description, param)
        path = self._resolve(description, creating=creating)
        return path_ref(path) if isinstance(path, str) else path

    def _resolve(self, d: Any, *, creating: bool) -> str | Unknown:
        if isinstance(d, Entity) and d.ref is not None and path_of(d.ref):
            return path_of(d.ref)
        if not isinstance(d, Entity):
            return Unknown("cannot_refer", f"not a description: {d!r}")
        if d.kind == "path":
            text = d.text.strip("'\"")
            if text.startswith("~"):
                return HOME + text[1:]
            if text.startswith("/"):
                return text
            return self._find(text) if not creating else f"{HOME}/{text}"
        noun = (d.features.get("noun") or d.text or "").lower()
        name = d.features.get("name")
        location = d.features.get("location")
        if name is None:
            # "the recipes folder": the words before the head noun name it
            said = [w for w in d.text.split() if w.lower() not in (noun, noun + "s", "the", "a", "an", "my", "your")]
            name = " ".join(said) or None
        for said in (d.text.lower(), noun):
            if said in self.places and name is None:
                return self.places[said]
        parent = self._resolve(location, creating=False) if location is not None else None
        if isinstance(parent, Unknown):
            return parent
        if name is not None:
            base = (name.text if isinstance(name, Entity) else str(name)).strip("'\"")
            if creating:
                return f"{parent or HOME}/{base}"
            return self._find(base, within=parent)
        if creating:
            return Unknown("unnamed", f"what should the new {noun} be called?")
        return self._find(noun, within=parent)

    def _find(self, name: str, within: str | None = None, depth: int = 3) -> str | Unknown:
        """Entries named ``name`` (or ``name.<ext>``) under ``within``, by listing, not by a search command."""
        hits, frontier = [], [(within or HOME, 0)]
        want = name.lower()
        while frontier:
            d, level = frontier.pop(0)
            for entry, is_dir in self.listing(d):
                low = entry.lower()
                if low == want or low.split(".")[0] == want:
                    hits.append(f"{d}/{entry}")
                if is_dir and level + 1 < depth:
                    frontier.append((f"{d}/{entry}", level + 1))
        if len(hits) == 1:
            return hits[0]
        if not hits:
            return Unknown("not_found", f"there is nothing called {name!r} " + (f"in {within}" if within else "under home"))
        return Unknown("ambiguous", f"{name!r} could be any of " + ", ".join(hits[:4]))

    def app_for(self, d: Any) -> Any | Unknown:
        if not isinstance(d, Entity):
            return Unknown("cannot_refer", "not a description")
        said = d.text.lower()
        match = [n for n in self.apps if n == said or said in n.split() or n in said]
        if len(match) == 1:
            return match[0]
        if not match:
            return Unknown("no_such_app", f"there is no {d.text} on this machine; its apps are " + ", ".join(sorted(self.apps)))
        return Unknown("ambiguous", f"{d.text!r} could be " + " or ".join(match))

    def execute(self, act: Call, *, key: str | None) -> Receipt:
        a = {k: path_of(v) or v for k, v in act.args}
        q = shlex.quote
        commands = {
            "make_directory": lambda: f"mkdir -p {q(a['path'])}",
            "make_file": lambda: f"touch {q(a['path'])}",
            "delete": lambda: f"rm -r {q(a['path'])}",
            "move": lambda: f"mv {q(a['path'])} {q(a['destination'] + '/' + a['path'].rsplit('/', 1)[-1])}",
            "list_directory": lambda: f"ls -1pA {q(a['directory'])}",
            "read_file": lambda: f"cat {q(a['path'])}",
        }
        if act.capability == "open_application":
            name = str(dict(act.args)["app"].id).split(":", 1)[1]
            control = self.apps.get(name)
            if control is None:
                return Receipt(act, "rejected", idempotency_key=key, error=f"no launcher for {name}")
            clicked = self.body.click(control)
            return Receipt(act, "applied" if clicked.status == "applied" else "failed", idempotency_key=key, error=clicked.error)
        if act.capability not in commands:
            return Receipt(act, "rejected", error=f"unknown capability {act.capability}")
        ok, out = self.run(commands[act.capability]())
        errors = [line for line in out if ": " in line and line.split(":", 1)[0] in ("mkdir", "touch", "rm", "mv", "ls", "cat")]
        if not ok:
            return Receipt(act, "failed", idempotency_key=key, error="the terminal did not take the command")
        if errors:
            return Receipt(act, "failed", idempotency_key=key, error=errors[0])
        self._last_output = out
        return Receipt(act, "applied", idempotency_key=key)

    def holds(self, cap: Capability, args: Mapping[str, Any]) -> bool | Unknown:
        p = {k: path_of(v) or v for k, v in args.items()}
        q = shlex.quote
        test = {
            "make_directory": lambda: (f"test -d {q(p['path'])} && echo yes || echo no", "yes"),
            "make_file": lambda: (f"test -f {q(p['path'])} && echo yes || echo no", "yes"),
            "delete": lambda: (f"test -e {q(p['path'])} && echo yes || echo no", "no"),
            "move": lambda: (f"test -e {q(p['destination'] + '/' + p['path'].rsplit('/', 1)[-1])} && echo yes || echo no", "yes"),
        }.get(cap.name)
        if cap.name == "open_application":
            from examples.browser_agents.perception.computerworld import windows_in

            name = str(args["app"].id).split(":", 1)[1]
            titles = [t.lower() for t, _ in windows_in(self.surface.scene()).values()]
            # a window is the app's if its title is the app's name or one of its words ("Editor")
            return any(t == name or t in name.split() or name in t for t in titles)
        if test is None:
            return Unknown("no_check", f"{cap.name} changes nothing to check")
        command, want = test()
        ok, out = self.run(command)
        if not ok or not out:
            return Unknown("unobserved", "the check printed nothing")
        return out[-1].strip() == want

    def reveal(self, cap: Capability, args: Mapping[str, Any], receipt: Receipt) -> Iterable[Claim]:
        if receipt.status != "applied":
            return
        out = getattr(self, "_last_output", [])
        if cap.name == "list_directory":
            d = path_of(args["directory"])
            for name, is_dir in self.listing(d):
                yield Claim(path_ref(f"{d}/{name}"), "has_location", path_ref(d))
                yield Claim(path_ref(f"{d}/{name}"), "is_a", "directory" if is_dir else "file")
        elif cap.name == "read_file":
            yield Claim(args["path"], "contain", "\n".join(out))


def _is_new(description: Any, param: Param) -> bool:
    """A description with a name is what a making-capability should create."""
    return isinstance(description, Entity) and description.features.get("name") is not None or (
        isinstance(description, Entity) and description.kind == "path")
