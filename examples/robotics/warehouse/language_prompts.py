"""Versioned proposal prompts; old experiment prompts remain immutable."""

from .language_review import V3_PROMPT

V4_PROMPT = (
    V3_PROMPT.replace('["parallel",MOVE,MOVE,...]', '["parallel",NODE,NODE,...]')
    .replace('["sequence",MOVE,MOVE,...]', '["sequence",NODE,NODE,...]')
    .replace(
        'MOVE is an array ["move","object reference","destination reference"].',
        'NODE is recursive: either ["move","object reference","destination reference"], '
        '["sequence",NODE,NODE,...], or ["parallel",NODE,NODE,...]. '
        "Composition nodes have at least two children. Nest NODE arrays as needed.",
    )
    + """
Output ONE JSON object with exactly ONE top-level key. NEVER repeat a JSON key.
A mixed sequential/concurrent request is one recursive instruction tree, not two
top-level instruction fields. All children of a sequence must finish in order;
children of a parallel node are independent. A sequence after a parallel group
begins only after every member of that group has finished.

Preserve EVERY reference modifier: color AND relation AND anchor identity.
Do not drop a color just because a spatial relation is also supplied. Do not pick
an ID for a descriptive reference; actual sensor grounding happens later.

User: Move parcel_10 to receiver_10 while moving parcel_11 to receiver_11; then move parcel_12 to receiver_12.
{"instruction":["sequence",["parallel",["move","parcel_10","receiver_10"],["move","parcel_11","receiver_11"]],["move","parcel_12","receiver_12"]]}
User: In parallel, perform these two branches: first move parcel_10 to receiver_10 then move parcel_11 to receiver_11; independently move parcel_12 to receiver_12.
{"instruction":["parallel",["sequence",["move","parcel_10","receiver_10"],["move","parcel_11","receiver_11"]],["move","parcel_12","receiver_12"]]}
User: Carry the green carton right of parcel_10 to receiver_11.
{"instruction":["move","green parcel right of parcel_10","receiver_11"]}
User: Where is the yellow package left of parcel_12?
{"where":"yellow parcel left of parcel_12"}
"""
)
