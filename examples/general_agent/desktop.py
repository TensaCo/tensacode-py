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


#: What the plugin knows how to do without trying: the desktop itself (its launcher and
#: its folders). Everything a command does is learned by running it (``discover.py``).
GUI_CAPABILITIES = (
    Capability("open_application", (Param("app", "application"),),
               effects=(Effect("has_state", {"undergoer": "app"}),), description="click its launcher button"),
    Capability("list_directory", (Param("directory", "directory"),),
               informs=(Informs("has_location", "goal", "directory"),), effect_kind="read", description="ls"),
)




class DesktopPlugin(Plugin):
    """Owns one actor session on a computerworld machine. Use from the thread that made it."""

    def __init__(self, world, *, on_step=None, learn: bool = True, learned: list | None = None) -> None:
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
        self.learned: list = []
        self._look_around()
        if learned is not None:
            # what was already found out about a machine of this kind. Discovery is an
            # experiment on the world, not a fact about this instance of it, so a caller
            # running many episodes on the same machine need not repeat it every time.
            self.learned = list(learned)
        elif learn:
            from examples.general_agent.discover import CANDIDATES, discover

            # what the commands on this machine do is found by trying them, on a snapshot
            self.learned = discover(self, world, CANDIDATES)

    # ------------------------------------------------------------------ the terminal

    def _shell(self):
        if not self.surface.has_terminal():
            self.surface.act("application.v1", "launch", {"kind": "terminal"})
        screen = self.body.observe()
        box = [c for c in screen.controls if c.role == "textbox" and c.name == "Shell input"]
        return box[0] if box else None

    def run(self, command: str) -> tuple[bool, list[str]]:
        """Type a command into the terminal and read what it printed.

        Whether it worked is the command's own exit status, which the terminal records per
        entry. It used to be inferred from the words in the output, and a shell that says
        "not found" in a sentence of its own is not the same as a shell that failed.
        """
        shell = self._shell()
        if shell is None:
            return False, ["no terminal"]
        before = len(self.surface.terminal_entries())
        receipt = self.body.fill(shell, command, submit=True)
        ran = self.surface.terminal_entries()[before:]
        out = [line for entry in ran for line in entry.lines()]
        self.log.append((command, out))
        worked = receipt.status == "applied" and all(e.ok is not False for e in ran)
        return worked, out

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
        return tuple(GUI_CAPABILITIES) + tuple(c for c, _ in self.learned)

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
        if isinstance(description, Entity) and not self.kind_fits(description, param.kind):
            # The agent asks the plugin before consulting its taxonomy, so the plugin has the
            # last word on its own kinds and has to use it: without this, `mkdir` accepted "a
            # file called draft.txt" and made a *directory* with that name, which then could
            # not be deleted ("rm: Is a directory").
            return Unknown("wrong_kind", f"{description.text} is not a {param.kind} here")
        creating = param.kind in ("directory", "file") and _is_new(description, param)
        path = self._resolve(description, creating=creating)
        if isinstance(path, str) and param.kind in ("path", "file"):
            # "move it to documents": a command that wants a full target path gets the
            # directory plus what is being moved, because that is what the name denotes here
            moving = [v for k, v in (context.get("args") or {}).items() if path_of(v)]
            if moving and self._is_directory(path):
                path = f"{path}/{path_of(moving[0]).rsplit('/', 1)[-1]}"
        return path_ref(path) if isinstance(path, str) else path

    def kind_fits(self, description: Entity, want: str) -> bool:
        """Is what this phrase names the kind of thing the parameter wants, by *this* machine's
        links: a folder is a directory, a file is a path, a directory is a path.

        A noun this desktop has no opinion about is allowed through — the world decides then,
        and refusing on silence would rule out every file name.
        """
        noun = str(description.features.get("noun") or "").lower()
        if not noun or noun == want:
            return True
        seen, frontier = {noun}, [noun]
        while frontier:
            here = frontier.pop()
            for up in self.kinds.get(here, ()):
                if up == want:
                    return True
                if up not in seen:
                    seen.add(up)
                    frontier.append(up)
        return noun not in self.kinds  # nothing known about it: let the world decide

    def _is_directory(self, path: str) -> bool:
        _, out = self.run(f"test -d {shlex.quote(path)} && echo d || echo f")
        return bool(out) and out[-1].strip() == "d"

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
        base = (name.text if isinstance(name, Entity) else str(name)).strip("'\"") if name is not None else None
        if creating:
            if base is None:
                return Unknown("unnamed", f"what should the new {noun} be called?")
            return f"{parent or HOME}/{base}"
        # Which word of the phrase identifies the thing is not something to decide in advance:
        # in "the file scratch.txt" the head noun is the file name and the modifier says what
        # kind it is, and in "the recipes folder" it is the other way round. So every identifier
        # the phrase offers is tried against the machine, and the one the machine actually has
        # is the referent. Committing to the modifier looked up "/home/agent/file" and deleted
        # nothing.
        tried: list[str] = []
        for candidate in [base, noun, *(w.strip("'\"") for w in d.text.split())]:
            if not candidate or candidate.lower() in ("the", "a", "an", "my", "your", "our") or candidate in tried:
                continue
            tried.append(candidate)
            got = self._find(candidate, within=parent)
            if not isinstance(got, Unknown):
                return got
        return Unknown("not_found", f"there is nothing called {' or '.join(repr(x) for x in tried)}"
                                    f"{' under ' + parent if parent else ' under home'}")

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
        if act.capability == "open_application":
            name = str(dict(act.args)["app"].id).split(":", 1)[1]
            control = self.apps.get(name)
            if control is None:
                return Receipt(act, "rejected", idempotency_key=key, error=f"no launcher for {name}")
            clicked = self.body.click(control)
            return Receipt(act, "applied" if clicked.status == "applied" else "failed", idempotency_key=key, error=clicked.error)
        if act.capability == "list_directory":
            self._last_output = [name for name, _ in self.listing(path_of(dict(act.args)["directory"]) or HOME)]
            return Receipt(act, "applied", idempotency_key=key)
        cap, template = self.command_for(act.capability)
        if template is None:
            return Receipt(act, "rejected", error=f"unknown capability {act.capability}")
        args = {k: path_of(v) or v for k, v in act.args}
        ok, out = self.run(template.format(*(shlex.quote(str(args[p.name])) for p in cap.params)))
        errors = [line for line in out if line.lower().startswith((act.capability + ":", "error"))]
        if not ok:
            return Receipt(act, "failed", idempotency_key=key, error="the terminal did not take the command")
        if errors:
            return Receipt(act, "failed", idempotency_key=key, error=errors[0])
        self._last_output = out
        return Receipt(act, "applied", idempotency_key=key)

    def command_for(self, name: str) -> tuple[Capability | None, str | None]:
        for cap, template in self.learned:
            if cap.name == name:
                return cap, template
        return None, None

    def holds(self, cap: Capability, args: Mapping[str, Any]) -> bool | Unknown:
        """Check the capability's *declared effects* against a fresh look, whatever they are.

        Nothing here is per-command: ``be`` means the path is there now, a negated
        ``has_location`` means it is not, and ``has_location`` with a goal means it is
        inside that goal. The same code checks a capability learned tomorrow.
        """
        if not cap.effects:
            return Unknown("no_check", f"{cap.name} declares no observable effects")
        checks = []
        unsupported = []
        for effect in cap.effects:
            # Checking a predicate requires checking all its roles.
            roles = set(effect.roles)
            target = args.get(effect.roles.get("undergoer", ""))
            if (cap.name == "open_application" and effect.pred == "has_state"
                    and roles == {"undergoer"} and not effect.negated
                    and isinstance(target, Ref) and target.id.startswith("app:")):
                from examples.browser_agents.perception.computerworld import windows_in

                name = target.id.split(":", 1)[1]
                titles = [t.lower() for t, _ in windows_in(self.surface.scene()).values()]
                if not any(t == name or t in name.split() or name in t for t in titles):
                    return False
                continue
            path = path_of(target)
            observed_path = None
            want = "yes"
            if path is not None:
                if roles == {"undergoer"} and effect.pred in ("be", "destroyed"):
                    observed_path = path
                    want = "yes" if effect.negated == (effect.pred == "destroyed") else "no"
                elif effect.pred == "has_location" and roles == {"undergoer"} and effect.negated:
                    observed_path, want = path, "no"
                elif effect.pred == "has_location" and roles == {"undergoer", "goal"}:
                    observed_path = _under(path_of(args.get(effect.roles["goal"])), path)
                    want = "no" if effect.negated else "yes"
            if observed_path is None:
                unsupported.append(effect.pred)
            else:
                checks.append((f"test -e {shlex.quote(observed_path)} && echo yes || echo no", want))
        for command, want in checks:
            ok, out = self.run(command)
            if not ok or not out or out[-1].strip() not in {"yes", "no"}:
                return Unknown("unobserved", "the check did not return an existence observation")
            if out[-1].strip() != want:
                return False
        if unsupported:
            return Unknown("unsupported_effect", "cannot verify complete effects: " + ", ".join(unsupported))
        return True

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


def _under(goal: str | None, moved: str) -> str | None:
    """Where the moved thing should be now: the goal itself if it already names it."""
    if not goal:
        return None
    base = moved.rsplit("/", 1)[-1]
    return goal if goal.rsplit("/", 1)[-1] == base else f"{goal}/{base}"


def _is_new(description: Any, param: Param) -> bool:
    """Whether the phrase is asking for something to be brought into existence.

    That is what definiteness is for: an indefinite phrase does not presuppose its referent
    ("make **a** folder called projects"), a definite one does ("delete **the** file
    scratch.txt"). Treating any phrase that carried a name as new made "delete the file
    scratch.txt" invent the path ``/home/agent/file`` and try to remove it, so the request
    ran and deleted nothing.

    Erring towards *existing* is the safe direction: a thing that turns out not to be there
    fails to resolve and the agent says so, whereas inventing a path acts on the wrong thing.
    """
    if not isinstance(description, Entity):
        return False
    if description.kind == "path":
        return True
    if description.kind == "resolved":
        # an anaphor presupposes its referent: "delete it" is about a thing already in the
        # conversation, whatever the phrase that introduced it happened to be ("create *a*
        # file …" then "delete it" must not create a second one)
        return False
    return description.features.get("definite") is False
