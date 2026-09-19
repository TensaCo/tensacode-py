"""Conversational formulas: what "hello" is, and what answers it.

Two different things live here, and only one of them is knowledge we had to write down.

**Which class an expression belongs to is WordNet's.** "hello", "hi" and "welcome" are a
``greeting``; "goodbye" is a ``farewell``; "thanks" is an ``acknowledgement``; "yes" is an
``affirmation``; "ok" is an ``approval``. Nothing here lists those words — the taxonomy is
asked, so "howdy" and "hullo" work for the same reason "hello" does, and so does any word
WordNet files under those classes.

**What answers what is convention, and conventions have to be seeded.** A greeting is
answered with a greeting, thanks with a formula that acknowledges it, a farewell with a
farewell. No corpus we have states that, and it differs between languages and registers, so
it is data with a source and a way to replace it — not a branch in the agent. ``PAIRS`` is
what a speaker of this register does; another register supplies another table.

What this deliberately does *not* do is guess. An expression whose class WordNet does not
give ("please", "sorry" — neither is a noun of the right kind) falls through and the agent
says it did not follow it, which is true.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping

#: The WordNet classes that make an utterance a conversational move rather than a statement
#: about the world. Reading the class from WordNet is what lets the words themselves be open.
CONVERSATIONAL = ("greeting", "farewell", "acknowledgement", "affirmation", "approval")

#: The reply a move of each class calls for — the second half of an adjacency pair. ``None``
#: means the move closes itself and no formula is owed ("ok" wants nothing back).
#:
#: Seeded, English, neutral register. Replaceable: set ``$TENSORCODE_CONVENTIONS`` to a JSON
#: file of the same shape, or hand ``pairs()`` a mapping of your own.
PAIRS: Mapping[str, str | None] = {
    "greeting": "hello",
    "farewell": "goodbye",
    "acknowledgement": "you are welcome",
    "affirmation": None,
    "approval": None,
}


def pairs(override: Mapping[str, str | None] | None = None) -> Mapping[str, str | None]:
    """The adjacency pairs in force: the seeded ones, a file's, or a caller's."""
    if override is not None:
        return dict(override)
    path = os.environ.get("TENSORCODE_CONVENTIONS")
    if path and Path(path).expanduser().is_file():
        return json.loads(Path(path).expanduser().read_text("utf-8"))
    return dict(PAIRS)


def move_of(kinds: frozenset[str] | set[str]) -> str | None:
    """Which conversational move an expression is, given what WordNet says it is a kind of.

    The most specific class wins, because ``greeting`` and ``farewell`` are both kinds of
    ``acknowledgement`` and answering "goodbye" with "hello" would be worse than saying
    nothing.
    """
    for name in CONVERSATIONAL:
        if name in kinds:
            return name
    return None
