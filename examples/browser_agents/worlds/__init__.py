"""World definitions for the computerworld engine (one JSON blueprint per kind of machine).

A world is data: profiles, computers with their initial files, a network and services. An
episode builds its definition, constructs ``World(definition, seed)`` and throws it away
afterwards — so episodes cannot contaminate each other, which the previous simulator could
not promise (its state outlived the run).
"""

from .desktop import DESKTOP_BASE, desktop_world, note_world  # noqa: F401
