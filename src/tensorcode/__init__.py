"""TensorCode: typed cognitive operations with swappable implementations.

Pre-alpha. The facades exported here (``parse``, ``classify``, ``choose``, ``rank``,
``check``, ``verify``, ``propose``, ``invoke``), the outcome values, records and the
runtime are the public API; submodules not re-exported here may change without notice.
"""

__version__ = "0.1.0a1"

from .actions import Plan, RunnablePlan, Step, action, invoke, plan_order, run_plan
from .context import Packed, approx_tokens, dedupe, pack, shingle_similarity
from .ops import Constraint, Objective, check, choose, classify, parse, propose, rank, verify
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
    "propose",
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
