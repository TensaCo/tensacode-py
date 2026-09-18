"""Run one probe suite against one assistant instance, and save the result.

    python -m eval.profile.run_probes <suite> <base-url> <out.json>

Each suite gets its own instance because a crashed request leaves the conversation
unrecoverable (see the ``copy`` defect in docs/revival/18-cognitive-profile.md), so
sharing a mind across suites would let one failure contaminate the next.
"""

from __future__ import annotations

import json
import sys

from . import probes
from .client import Chat

SUITES = {
    "compositionality": probes.compositionality,
    "belief_revision": probes.belief_revision,
    "grounding": probes.grounding,
}


def main() -> None:
    suite, base, out = sys.argv[1], sys.argv[2], sys.argv[3]
    chat = Chat(base=base)
    cases = SUITES[suite](chat)
    chat.close()
    rows = [dict(name=c.name, said=c.said, checks=c.checks, replies=c.replies, acts=c.acts,
                 seconds=round(c.seconds, 2), model_calls=c.model_calls, passed=c.passed) for c in cases]
    Path = __import__("pathlib").Path
    Path(out).write_text(json.dumps(rows, indent=1))
    for r in rows:
        print(f"{r['name']:42s} pass={str(r['passed']):5s} acts={r['acts']}")
        for k, v in r["checks"].items():
            print(f"    {'ok  ' if v else 'FAIL'} {k}")


if __name__ == "__main__":
    main()
