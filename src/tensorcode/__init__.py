"""TensorCode: typed cognitive operations with swappable implementations.

Pre-alpha. No part of the API is stable before 1.0: any 0.x release may change or remove
anything, and CHANGELOG.md lists what changed. The names exported here are the core the
design is built around; the other submodules are more experimental still.
"""

__version__ = "0.1.0a1"

from .actions import Plan, RunnablePlan, Step, action, invoke, plan_order, run_plan
from .context import Packed, approx_tokens, dedupe, pack, shingle_similarity
from .ops import Constraint, Objective, check, choose, classify, parse, rank, verify
from .outcomes import Receipt, Score, Unknown, Verdict
from .records import (
    Claim,
    Conflict,
    Evidence,
    Interval,
    Opaque,
    Patch,
    Ref,
    Retract,
    SetField,
    Store,
    Tell,
    TypeRegistry,
    Var,
)
from .runtime import Budget, Output, Policy, Profile, Request, Runtime, Trace, Traits, implementation, use

from . import actions, records

__all__ = [
    "Plan",
    "Step",
    "action",
    "plan_order",
    "RunnablePlan",
    "invoke",
    "run_plan",
    "Packed",
    "approx_tokens",
    "dedupe",
    "pack",
    "shingle_similarity",
    "Constraint",
    "Objective",
    "check",
    "choose",
    "classify",
    "parse",
    "rank",
    "verify",
    "Receipt",
    "Score",
    "Unknown",
    "Verdict",
    "Claim",
    "Conflict",
    "Evidence",
    "Interval",
    "Opaque",
    "Patch",
    "Ref",
    "Retract",
    "SetField",
    "Store",
    "Tell",
    "TypeRegistry",
    "Var",
    "Budget",
    "Output",
    "Policy",
    "Profile",
    "Request",
    "Runtime",
    "Trace",
    "Traits",
    "implementation",
    "use",
    "actions",
    "records",
]
