"""Desktop jobs written for this environment, graded by what the world looks like afterwards.

Why these exist: the assay's `computer_use.shell_files` items come from NL2Bash, and they are
sysadmin one-liners — `find ~ -atime +100 -delete`, `ps aux | awk …`, `sudo ln -s`, a suid
scan of `/`. They cannot be graded by running the reference command (one of them deletes the
home tree), most of them cannot run in a desktop world at all, and an agent that abstains on
them is behaving correctly. So that task measures engagement and nothing more.

These are jobs a person would actually ask a desktop assistant for, and each one says what
the world should look like when it is done. Grading reads the machine through the **owner**
session, never the agent's own report: "I made the folder" is not evidence that a folder
exists, and an agent that says so without doing it must score zero.

**This is a self-authored task set.** The prompts and the graders are both ours, so it is a
regression suite and never a headline number (`docs/revival/11`). Two things keep it from
flattering us: the prompts were written as English, not as input the parser is known to
handle, and nothing here was adjusted after seeing a score. Whatever fraction it reports is
the fraction — the failures are the point of having it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

HOME = "/home/agent"
DESKTOP = f"{HOME}/Desktop"


@dataclass(frozen=True)
class Job:
    """One request, with what has to be true afterwards.

    ``check`` is given the world (owner access, for grading) and the agent's reply. A job
    about the world ignores the reply; a job that asks a question is graded on the reply,
    because the answer *is* the deliverable.
    """

    id: str
    prompt: str
    check: Callable[[object, str], bool]
    setup: tuple[str, ...] = ()          # owner commands that put the machine in its start state
    about: str = ""                      # what this job is testing, for the report


def _exists(path: str) -> Callable[[object, str], bool]:
    return lambda world, reply: world.exists(path)


def _gone(path: str) -> Callable[[object, str], bool]:
    return lambda world, reply: not world.exists(path)


def _contains(path: str, text: str) -> Callable[[object, str], bool]:
    def check(world, reply: str) -> bool:
        got = world.read(path)
        return bool(got) and text.lower() in got.lower()

    return check


def _moved(source: str, destination: str) -> Callable[[object, str], bool]:
    return lambda world, reply: world.exists(destination) and not world.exists(source)


def _said(*words: str) -> Callable[[object, str], bool]:
    """The reply has to contain what was asked for. Used only for questions."""

    def check(world, reply: str) -> bool:
        low = (reply or "").lower()
        return all(w.lower() in low for w in words)

    return check


def _unchanged(path: str) -> Callable[[object, str], bool]:
    return lambda world, reply: world.exists(path)


JOBS: tuple[Job, ...] = (
    Job("make-folder", "Make a folder called projects on my desktop.",
        _exists(f"{DESKTOP}/projects"), about="one capability, one argument"),
    Job("make-nested", "Create a folder named drafts inside my documents folder.",
        _exists(f"{HOME}/Documents/drafts"), about="a place named by a possessive"),
    Job("create-file", "Create a file called todo.txt on my desktop.",
        _exists(f"{DESKTOP}/todo.txt"), about="a file rather than a directory"),
    Job("write-file", "Put the line 'buy milk' into a file called shopping.txt on my desktop.",
        _contains(f"{DESKTOP}/shopping.txt", "buy milk"),
        about="content, which needs the right capability and the text carried through"),
    Job("delete-file", "Delete the file scratch.txt from my desktop.",
        _gone(f"{DESKTOP}/scratch.txt"), setup=(f"touch {DESKTOP}/scratch.txt",),
        about="destruction, and referring to something that exists"),
    Job("move-file", "Move report.txt from my desktop into my documents folder.",
        _moved(f"{DESKTOP}/report.txt", f"{HOME}/Documents/report.txt"),
        setup=(f"echo draft > {DESKTOP}/report.txt", f"mkdir -p {HOME}/Documents"),
        about="two arguments, and the whole request rather than half of it"),
    Job("copy-file", "Copy notes.txt from my desktop into my documents folder.",
        lambda world, reply: world.exists(f"{HOME}/Documents/notes.txt") and world.exists(f"{DESKTOP}/notes.txt"),
        setup=(f"echo hello > {DESKTOP}/notes.txt", f"mkdir -p {HOME}/Documents"),
        about="copying keeps the original; deleting it would be half the job done wrong"),
    Job("append-line", "Add the line 'second' to the end of list.txt on my desktop.",
        _contains(f"{DESKTOP}/list.txt", "second"),
        setup=(f"echo first > {DESKTOP}/list.txt",),
        about="appending, which must not lose what was there"),
    Job("keep-first-line", "Add the line 'second' to the end of list.txt on my desktop.",
        _contains(f"{DESKTOP}/list.txt", "first"),
        setup=(f"echo first > {DESKTOP}/list.txt",),
        about="the same job, graded on what it must NOT destroy"),
    Job("list-desktop", "What is on my desktop?", _said("readme-first.txt"),
        about="a question answered by looking, not from memory"),
    Job("read-file", "What does hello.txt on my desktop say?", _said("greetings"),
        setup=(f"echo greetings > {DESKTOP}/hello.txt",),
        about="reading a file's contents back"),
    Job("refuse-unknown", "Encrypt my desktop with a quantum cipher.",
        _unchanged(f"{DESKTOP}/readme-first.txt"),
        about="a request nothing can serve: the world must be untouched"),
)
