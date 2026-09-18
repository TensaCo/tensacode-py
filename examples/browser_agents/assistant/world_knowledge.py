"""What the assistant knows before it looks: what apps are for, and what a topic is called.

This is data, not a lookup buried in a function. It enters the mind as claims with their own
provenance (``prior:apps``), so ``explain`` can say *why* it reached for Visual Studio Code
when asked for "the app used for writing code", and so a claim can be added, corrected or
contradicted by something seen on screen later.

Kept deliberately small: what is here is what a person would assume a desktop assistant
already knows. Anything else it has to look up or ask about.
"""

from __future__ import annotations

import tensorcode as tc

#: app -> the things it is for, in the words someone would actually use
APP_FUNCTIONS: dict[str, tuple[str, ...]] = {
    "Visual Studio Code": ("writing code", "programming", "software development", "editing source files", "coding"),
    "Text Editor": ("writing text", "editing a text file", "taking notes", "writing plain text"),
    "Terminal": ("running commands", "using the shell", "typing commands", "running a script"),
    "Files": ("browsing files", "looking at folders", "managing files", "finding a file"),
    "Firefox": ("browsing the web", "opening a website", "reading web pages", "searching the internet"),
    "Chromium": ("browsing the web", "opening a website", "web development"),
    "Mail": ("sending email", "reading email", "writing a message to someone"),
    "Slack": ("chatting with colleagues", "sending a message to a channel", "team chat"),
    "Rhythmbox": ("playing music", "listening to audio"),
    "System Monitor": ("watching cpu and memory", "seeing what is running", "checking system load"),
    "Settings": ("changing settings", "configuring the machine", "changing preferences"),
    "App Center": ("installing software", "finding new applications"),
    "Wireshark": ("capturing network traffic", "inspecting packets", "watching the network"),
}

#: how people refer to the same app
APP_ALIASES: dict[str, tuple[str, ...]] = {
    "Visual Studio Code": ("vs code", "vscode", "code editor", "ide"),
    "Text Editor": ("gedit", "notepad", "text edit"),
    "Terminal": ("console", "shell", "command line", "bash"),
    "Files": ("file manager", "explorer", "nautilus", "finder"),
    "Firefox": ("browser", "web browser"),
    "Mail": ("email", "email client", "mail client"),
    "System Monitor": ("task manager", "activity monitor"),
}


def claims() -> list[tc.Claim]:
    """Prior knowledge as claims, ready to be integrated into a fresh mind."""
    out: list[tc.Claim] = []
    for app, functions in APP_FUNCTIONS.items():
        ref = tc.Ref(f"app:{app}")
        out.append(tc.Claim(ref, "is_a", "application"))
        out.append(tc.Claim(ref, "name", app))
        out += [tc.Claim(ref, "is_for", function) for function in functions]
    for app, aliases in APP_ALIASES.items():
        out += [tc.Claim(tc.Ref(f"app:{app}"), "also_called", alias) for alias in aliases]
    return out
