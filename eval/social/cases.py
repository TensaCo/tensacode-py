"""Cases for the four social faculties, each labelled with where it came from.

Provenance matters more than count here. ``USER`` rows are the owner's own words, taken from
the failing transcript and from limits recorded in docs/revival/15 — those are the acceptance
set, and nothing may be special-cased to pass them. ``FLOOR`` rows I wrote myself, split into
a design half (visible while building) and a held-out half, and they measure obvious breakage
rather than coverage: a set written beside the system inherits the system's blind spots.
"""

from __future__ import annotations

from dataclasses import dataclass, field

USER = "user (own transcript / recorded limit)"
FLOOR = "floor (authored here)"


@dataclass(frozen=True)
class Case:
    text: str
    want: str  # the act expected, or "unknown" / "no_action"
    provenance: str
    half: str = "design"  # design | heldout
    slots: dict = field(default_factory=dict)
    note: str = ""


# ---------------------------------------------------------------- indirect
# A request wearing another form. "no_action" rows are near misses that must NOT become acts:
# a gain on the requests bought with a loss here is a failure, so both are reported.

INDIRECT = [
    # the owner's own recorded example (docs/revival/15: read as a command by shape)
    Case("it would be good if you deleted notes.txt", "delete", USER, slots={"target": "notes.txt"}),
    Case("I can't find my invoice", "find", USER, slots={"pattern": "invoice"}),
    Case("I can't find my keys", "no_action", USER, note="not my domain: the gate must refuse this"),
    # design half
    Case("it'd be nice if you renamed notes.txt to ideas.txt", "rename", FLOOR, slots={"target": "notes.txt"}),
    Case("I'd like you to read notes.txt", "read", FLOOR, slots={"target": "notes.txt"}),
    Case("I'm looking for my budget.xlsx", "find", FLOOR, slots={"pattern": "budget.xlsx"}),
    Case("where did my report.pdf go", "find", FLOOR, slots={"pattern": "report.pdf"}),
    Case("is there a readme?", "find", FLOOR, slots={"pattern": "readme"}),
    Case("my desktop is a mess", "clarify_goal", FLOOR, slots={"place": "~/Desktop"}),
    Case("I can't find my wallet", "no_action", FLOOR, note="not my domain"),
    Case("I'm looking for a new job", "no_action", FLOOR, note="not my domain"),
    Case("I wish I had more time", "no_action", FLOOR, note="a wish about nothing I can act on"),
    Case("it would be good if it stopped raining", "no_action", FLOOR, note="wish, object outside my reach"),
    # held-out half
    Case("would you mind deleting old.log", "delete", FLOOR, "heldout", {"target": "old.log"}),
    Case("if you could copy budget.xlsx to documents that would help", "copy", FLOOR, "heldout", {"target": "budget.xlsx"}),
    Case("I need you to open firefox", "open_app", FLOOR, "heldout", {"app": "Firefox"}),
    Case("I can't find the shopping list", "find", FLOOR, "heldout", {"pattern": "shopping list"}),
    Case("where did my screenshots go", "find", FLOOR, "heldout", {"pattern": "screenshots"}),
    Case("do I have a backup?", "find", FLOOR, "heldout", {"pattern": "backup"}),
    Case("my downloads folder is a disaster", "clarify_goal", FLOOR, "heldout", {"place": "~/Downloads"}),
    Case("I can't find my glasses", "no_action", FLOOR, "heldout", note="not my domain"),
    Case("I'm looking for my car", "no_action", FLOOR, "heldout", note="not my domain"),
    Case("it would be good if you were faster", "no_action", FLOOR, "heldout", note="about me, not a file"),
    Case("I wish this machine had more memory", "no_action", FLOOR, "heldout", note="not actionable by me"),
    Case("the weather is a mess", "no_action", FLOOR, "heldout", note="a mess, but not a folder"),
]

# ------------------------------------------------------------ clarification
# Underdetermined goals (should ask) against clear requests (must not ask). The second rate is
# the one that makes an assistant tiresome, so it is reported beside the first.

UNDERDETERMINED = [
    Case("organize my desktop", "clarify_goal", USER, note="recorded as a flat refusal before this work"),
    Case("clean up my desktop", "clarify_goal", FLOOR),
    Case("tidy up documents", "clarify_goal", FLOOR),
    Case("sort out my downloads", "clarify_goal", FLOOR, "heldout"),
    Case("straighten up ~/Projects", "clarify_goal", FLOOR, "heldout", {"place": "~/Projects"}),
    Case("organise my pictures", "clarify_goal", FLOOR, "heldout"),
]

CLEAR = [  # must be acted on, never questioned
    Case("whats on my desktop", "list", USER),
    Case("read notes.txt", "read", USER),
    Case("make a folder called recipes on my desktop", "create_folder", USER),
    Case("delete notes.txt", "delete", FLOOR),
    Case("how many icons are in the sidebar", "ask_screen", USER),
    Case("what is my name", "ask_memory", USER),
    Case("find all pdfs", "find", FLOOR, "heldout"),
    Case("copy notes.txt to documents", "copy", FLOOR, "heldout"),
    Case("what time is it", "info", FLOOR, "heldout"),
    Case("open firefox", "open_app", FLOOR, "heldout"),
]

# -------------------------------------------------------- false presupposition
# A request that takes something for granted which is not so. The reply should correct the
# belief and name the near miss, not report a bare absence.

PRESUPPOSED = [
    Case("delete the report.txt", "correct", USER, slots={"near": "reports.txt"}, note="owner's own prompt"),
    Case("read notse.txt", "correct", FLOOR, slots={"near": "notes.txt"}, note="transposition"),
    Case("delete recipe", "correct", FLOOR, slots={"near": "recipes"}, note="singular for a plural folder"),
    # reclassified after measuring: the assistant finds this one recursively and names the path
    # it read, which *is* the belief repair. "did you mean" was the wrong expectation, not the
    # wrong behaviour — recorded in docs/revival/24 rather than quietly dropped.
    Case("read pasta.txt", "located", FLOOR, "heldout", {"at": "~/Desktop/recipes"},
         note="right name, wrong place: the reply must say where it really is"),
    Case("delete notes.text", "correct", FLOOR, "heldout", {"near": "notes.txt"}, note="wrong extension"),
    Case("read shoping.txt", "correct", FLOOR, "heldout", {"near": "shopping.txt"}, note="misspelling"),
    Case("organise my videos", "no_folder", FLOOR, "heldout",
         note="presupposes a folder that is not there; calling it empty would be a false statement"),
    Case("read zzzqqq.txt", "absent", FLOOR, "heldout", note="nothing near: a bare absence is the honest reply"),
]
