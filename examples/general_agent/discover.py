"""Learn what commands do by trying them, instead of declaring it in code.

The desktop plugin used to say, in Python, that ``mkdir`` makes a directory exist and
``rm`` destroys it. That is knowledge about the world written into the program. Here the
plugin finds it out the way anyone would: set up a scene it knows, run the command, look
at what changed, and put the change in the same predicates a request becomes.

* **what to try** comes from tldr-pages (CC BY 4.0): the command's name and its argument
  shapes (``{{path/to/directory}}``, ``{{path/to/source}}``), as data;
* **the trying** happens on a snapshot of the world, restored afterwards, so discovery
  leaves nothing behind;
* **the looking** is the actor's own listing of the machine, not a privileged read (only
  the *setup* of the probe scene uses the owner session, as any test harness would);
* **whether it ran** is the command's exit status, which the terminal reports per entry —
  not a search of the output for "not found", which is a sentence a working command can
  print;
* **the induced effect** is ``be`` for something that appeared, ``destroyed`` and
  ``not has_location`` for something gone, ``has_location`` for something that moved, and
  no content effect for changed text until a content binding can be learned.

A command that errors, or changes nothing, yields no capability: this machine's shell is
not the one tldr describes, and what it actually supports is what the agent finds.
"""

from __future__ import annotations

import os
import shlex
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence

from tensorcode.agent.plugin import Capability, Effect, Informs, Param

HOME = "/home/agent"

#: The commands to try on a desktop. tldr describes thousands; the machine decides which
#: of these it actually supports.
CANDIDATES = ("mkdir", "touch", "rm", "mv", "cp", "cat", "echo", "grep", "head", "tail", "wc")
PROBE = f"{HOME}/.probe"

#: What a tldr placeholder is asking for, by the words it uses. Reading the placeholder is
#: reading tldr's data; these are its own words.
KIND_OF_PLACEHOLDER = (("existing_directory", "directory"), ("directory", "directory"), ("target", "path"),
                       ("source", "path"), ("file", "file"), ("path", "path"))


def find_tldr() -> Path | None:
    for c in (os.environ.get("TENSORCODE_TLDR"), "~/.cache/tensorcode/seeds/tldr/pages"):
        if c and Path(c).expanduser().is_dir():
            return Path(c).expanduser()
    return None


@dataclass(frozen=True)
class Usage:
    """One command line from tldr, with its placeholders."""

    command: str
    template: str            # e.g. "mv {0} {1}"
    slots: tuple[str, ...]   # the placeholder text of each {i}
    description: str

    def kinds(self) -> tuple[str, ...]:
        out = []
        for slot in self.slots:
            low = slot.lower()
            out.append(next((kind for word, kind in KIND_OF_PLACEHOLDER if word in low), "path"))
        return tuple(out)


def usages(command: str, root: Path | None = None, *, limit: int = 8) -> list[Usage]:
    """The usage lines tldr gives for a command, as fillable templates.

    Enough of them to reach the ones that differ in kind ("rm file" and "rm -r directory"),
    since which of them this machine supports is decided by trying.
    """
    root = root or find_tldr()
    if root is None:
        return []
    page = next((p for p in (root / "common" / f"{command}.md", root / "linux" / f"{command}.md") if p.exists()), None)
    if page is None:
        return []
    out: list[Usage] = []
    description = ""
    for line in page.read_text("utf-8").splitlines():
        line = line.strip()
        if line.startswith("- "):
            description = line[2:].rstrip(":")
        elif line.startswith("`") and line.endswith("`"):
            got = _usage(command, line.strip("`"), description)
            if got is not None:
                out.append(got)
    return out[:limit]


def _literal(text: str) -> str:
    """Fixed text of a usage line, safe to put in a format template."""
    return text.replace("{", "{{").replace("}", "}}")


def _usage(command: str, body: str, description: str) -> Usage | None:
    """One tldr line as a fillable template.

    tldr writes a flag choice as ``{{[-p|--parents]}}`` (take the short form: it is fixed
    text, not an argument) and a repeatable argument as ``{{path/to/dir1 path/to/dir2 ...}}``
    (one is enough to see what the command does). Split on the braces; no pattern matching.
    """
    if not body.startswith(command + " "):
        return None
    slots: list[str] = []
    out: list[str] = []
    rest = body
    while "{{" in rest:
        before, rest = rest.split("{{", 1)
        if "}}" not in rest:
            return None
        inner, rest = rest.split("}}", 1)
        # a command's own braces are literal text, not a slot: awk programs are written
        # "{print $5}" and a template built from them must not be read as a format field
        out.append(_literal(before))
        inner = inner.strip()
        if inner.startswith("[") and inner.endswith("]"):
            out.append(inner[1:-1].split("|")[0])
            continue
        words = inner.replace("...", "").split()
        if not words:
            return None
        name = words[0]
        if "{" in name or "}" in name or name.startswith("/dev/"):
            return None
        slots.append(name)
        out.append("{%d}" % (len(slots) - 1))
    out.append(_literal(rest))
    template = "".join(out)
    if not slots or "|" in template:
        return None
    return Usage(command, " ".join(template.split()), tuple(slots), description)


# ------------------------------------------------------------------ the experiment


@dataclass(frozen=True)
class Scene:
    """What the machine looked like: every path under home, with its kind and text."""

    entries: dict[str, str]           # path -> "dir" | "file"
    text: dict[str, str]              # path -> contents, for files small enough to read

    def diff(self, later: "Scene") -> dict[str, list[str]]:
        gone = [p for p in self.entries if p not in later.entries]
        new = [p for p in later.entries if p not in self.entries]
        changed = [p for p in self.text if p in later.text and later.text[p] != self.text[p]]
        return {"gone": sorted(gone), "new": sorted(new), "changed": sorted(changed)}


def observe(plugin, root: str = HOME, depth: int = 3) -> Scene:
    """What the agent can see of the machine, through its own terminal."""
    entries: dict[str, str] = {}
    text: dict[str, str] = {}
    frontier = [(root, 0)]
    while frontier:
        directory, level = frontier.pop()
        for name, is_dir in plugin.listing(directory):
            path = f"{directory}/{name}"
            entries[path] = "dir" if is_dir else "file"
            if is_dir and level + 1 < depth:
                frontier.append((path, level + 1))
            elif not is_dir:
                ok, out = plugin.run(f"cat {shlex.quote(path)}")
                if ok:
                    text[path] = "\n".join(out)
    return Scene(entries, text)


def probe_values(kinds: Sequence[str], world) -> list[str]:
    """Concrete arguments for one trial: existing things for inputs, a fresh name for outputs."""
    values = []
    for i, kind in enumerate(kinds):
        if kind == "directory":
            values.append(f"{PROBE}/dir{i}")
        elif kind == "file":
            values.append(f"{PROBE}/file{i}.txt")
        else:
            values.append(f"{PROBE}/thing{i}")
    return values


def setup_scene(world, values: Sequence[str], *, last_exists: bool) -> None:
    """Create what a trial needs. Setup is the harness's job, so it uses the owner session.

    Whether the *last* argument should already exist is not something to guess: ``rm``
    needs its file to be there and ``mkdir`` needs its directory not to be, so both are
    tried and the machine decides which makes sense.
    """
    world.shell(f"mkdir -p {PROBE}")
    for v in values if last_exists else values[:-1]:
        if v.endswith(".txt"):
            world.shell(f"echo probe > {v}")
        else:
            world.shell(f"mkdir -p {v}")
            world.shell(f"echo probe > {v}/inside.txt")
    if values:
        world.shell(f"mkdir -p {os.path.dirname(values[-1])}")


def effects_from(diff: dict[str, list[str]], values: Sequence[str], names: Sequence[str]) -> tuple[list[Effect], list[Informs]]:
    """The representable change, in the predicates a request becomes.

    New paths establish existence only. Neither new nor changed text establishes
    a usable ``contain`` effect: this interface has no content-valued parameter
    binding. A unary ``contain`` would claim success without saying which
    contents were produced. Scene.text retains the observations for future
    content-aware induction.
    """
    of_value = {v: n for n, v in zip(names, values)}
    effects: list[Effect] = []
    informs: list[Informs] = []
    for path in diff["new"]:
        name = of_value.get(path)
        if name:
            effects.append(Effect("be", {"undergoer": name}))
    for path in diff["gone"]:
        name = of_value.get(path)
        if name:
            effects.append(Effect("has_location", {"undergoer": name}, negated=True))
            effects.append(Effect("destroyed", {"undergoer": name}))
    moved = [p for p in diff["new"] if any(p.rsplit("/", 1)[-1] == g.rsplit("/", 1)[-1] for g in diff["gone"])]
    if moved and len(values) >= 2:
        # a thing that vanished here and appeared there moved: it was not destroyed
        effects = [e for e in effects if e.pred not in ("be", "destroyed")]
        effects.append(Effect("has_location", {"undergoer": names[0], "goal": names[-1]}))
    return effects, informs


def widen_kinds(plugin, world, cap: Capability, usage: Usage) -> Capability:
    """What a slot accepts is also learned by trying: ``rm`` takes a directory too, whatever
    the manual's placeholder happened to call it."""
    params = list(cap.params)
    for i, param in enumerate(params):
        if param.kind != "file":
            continue
        values = [f"{PROBE}/dir{j}" if j == i else v for j, v in enumerate(probe_values([p.kind for p in params], world))]
        snapshot = world.snapshot()
        try:
            setup_scene(world, values, last_exists=True)
            before = observe(plugin)
            ok, out = plugin.run(usage.template.format(*(shlex.quote(v) for v in values)))
            worked = ok and any(before.diff(observe(plugin)).values())
        finally:
            world.restore(snapshot)
        if worked:
            params[i] = Param(param.name, "path", param.role)
    return replace(cap, params=tuple(params))


def discover(plugin, world, commands: Iterable[str]) -> list[tuple[Capability, str]]:
    """Try each command on a snapshot and keep the ones whose effect can be seen.

    Returns each capability with the command template that produced it, so the plugin can
    run later what it ran while learning.
    """
    found: list[tuple[Capability, str]] = []
    signatures: set = set()
    for command in commands:
        for usage in usages(command):
            kinds = usage.kinds()
            names = [f"arg{i}" for i in range(len(kinds))]
            values = probe_values(kinds, world)
            learned = None
            for last_exists in (False, True):
                snapshot = world.snapshot()
                try:
                    setup_scene(world, values, last_exists=last_exists)
                    before = observe(plugin)
                    ok, out = plugin.run(usage.template.format(*(shlex.quote(v) for v in values)))
                    if not ok:
                        continue
                    diff = before.diff(observe(plugin))
                    effects, informs = effects_from(diff, values, names)
                    if not effects and any(diff.values()):
                        # An observed mutation with no representable effect is not a read.
                        continue
                    if not effects and not out:
                        continue
                    if out and not effects:
                        informs = [Informs("contain", "undergoer", names[0])]
                    learned = Capability(command, tuple(Param(n, k) for n, k in zip(names, kinds)),
                                         tuple(effects), tuple(informs),
                                         effect_kind="read" if not effects else "write",
                                         description=f"{usage.description} (learned by trying it)")
                finally:
                    world.restore(snapshot)
                if learned is not None:
                    break
            if learned is not None:
                learned = widen_kinds(plugin, world, learned, usage)
                signature = (tuple(sorted((e.pred, e.negated, tuple(sorted(e.roles.items()))) for e in learned.effects)),
                             tuple(p.kind for p in learned.params), tuple(sorted(i.pred for i in learned.informs)))
                if signature in signatures:  # same effect on the same kinds: one way of saying it is enough
                    continue
                signatures.add(signature)
                # every usage that works is its own capability: "rm file" and "rm -r directory"
                # do different things to different kinds, and the planner picks by kind
                seen = sum(1 for c, _ in found if c.name.split("#")[0] == command)
                if seen:
                    learned = replace(learned, name=f"{command}#{seen + 1}")
                found.append((learned, usage.template))
    return found
