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

Execution consumes explicit application identities and absolute paths. This adapter does
not resolve descriptions, search names, infer destinations, or classify requested nouns.
"""

from __future__ import annotations

import shlex
from typing import Any, Iterable, Mapping

from examples.browser_agents.browser import is_computerworld_terminal_input
from examples.browser_agents.perception.computerworld import CwProvider
from examples.browser_agents.perception.cw_body import CwBody
from tensorcode.agent.plugin import Call, Capability, Effect, Informs, Param, Plugin
from tensorcode.language import words
from tensorcode.outcomes import Receipt, Unknown
from tensorcode.records import Proposition, Var, Claim, Ref

HOME = "/home/agent"


def path_ref(path: str) -> Ref:
    return Ref(f"path:{path}")


def path_of(ref: Any) -> str | None:
    if isinstance(ref, Ref) and ref.id.startswith("path:/") and "\x00" not in ref.id:
        return ref.id[len("path:"):]
    return None


#: What the plugin knows how to do without trying: the desktop itself (its launcher and
#: its folders). Everything a command does is learned by running it (``discover.py``).
GUI_CAPABILITIES = (
    Capability("open_application", (Param("app", "application"),),
               effects=(Effect("has_state", {"undergoer": "app"}),), description="click its launcher button"),
    Capability("list_directory", (Param("directory", "directory"),),
               informs=(Informs("has_location", "goal", "directory",
                   query=Proposition("has_location", {"subject": Var("answer"), "object": Var("directory")})),), effect_kind="read", description="ls"),
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
        box = [c for c in screen.controls if is_computerworld_terminal_input(c)]
        return box[0] if len(box) == 1 else None

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
        worked = receipt.status == "applied" and bool(ran) and all(e.ok is not False for e in ran)
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

    def execute(self, act: Call, *, key: str | None) -> Receipt:
        if act.plugin != self.name:
            return Receipt(act, "rejected", error="call belongs to a different provider")
        cap = next((c for c in self.capabilities() if c.name == act.capability), None)
        if cap is None:
            return Receipt(act, "rejected", error=f"unknown capability {act.capability}")
        args = dict(act.args)
        if len(args) != len(act.args) or set(args) != {p.name for p in cap.params}:
            return Receipt(act, "rejected", error="arguments must exactly match declared parameters")
        for param in cap.params:
            value = args[param.name]
            if param.kind in ("path", "file", "directory"):
                valid = path_of(value) is not None
            elif param.kind == "application":
                valid = isinstance(value, Ref) and value.id.startswith("app:") and value.id[4:] in self.apps
            else:
                valid = isinstance(value, str) and "\x00" not in value
            if not valid:
                return Receipt(act, "rejected", error=f"invalid explicit {param.kind} argument: {param.name}")
        if act.capability == "open_application":
            clicked = self.body.click(self.apps[args["app"].id[4:]])
            return Receipt(act, "applied" if clicked.status == "applied" else "indeterminate",
                           idempotency_key=key, error=clicked.error)
        if act.capability == "list_directory":
            directory = path_of(args["directory"])
            ok, out = self.run(f"test -d {shlex.quote(directory)} && ls -1A {shlex.quote(directory)}")
            if not ok:
                return Receipt(act, "failed", error="explicit directory could not be listed")
            self._last_listing = tuple(out)
            return Receipt(act, "applied", idempotency_key=key)
        _, template = self.command_for(act.capability)
        if template is None:
            return Receipt(act, "rejected", error=f"unknown command template {act.capability}")
        values = [path_of(args[p.name]) if p.kind in ("path", "file", "directory") else args[p.name]
                  for p in cap.params]
        try:
            command = template.format(*(shlex.quote(value) for value in values))
        except (IndexError, KeyError, ValueError) as exc:
            return Receipt(act, "rejected", error=f"invalid command model: {exc}")
        ok, out = self.run(command)
        if not ok:
            return Receipt(act, "indeterminate", idempotency_key=key,
                           error="command did not complete successfully; effects may be partial")
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
                    observed_path = path_of(args.get(effect.roles["goal"]))
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
            if d is None:
                return
            for name in getattr(self, "_last_listing", ()):
                yield Claim(path_ref(f"{d.rstrip('/')}/{name}"), "has_location", path_ref(d))
        elif cap.name == "read_file":
            yield Claim(args["path"], "contain", "\n".join(out))
