"""Explicit application infrastructure, separate from pretrained model tools.

These utilities execute supplied policies and manage state. They do not load
models, initialize cognitive weights, or provide learned competence themselves.
"""
from .action_loop import (
    ActionLoop, ActionLoopResult, ActionOutcome, ActionReceipt, ActionRequest,
)
from .memory import JsonMemory, MemoryRecord, MemorySearch
from .message_memory import decode_message_sequence, encode_message_sequence
from .pipeline import DecisionPipeline
from .cognitive_state import CognitiveState, Evidence, Hypothesis, Assessment, Goal, Plan, Observation
from .episodic import EpisodicMemory, RetrievalHit
from .planning import (
    PlanStep, ExecutablePlan, OutcomeExperience, ReplanRequest,
    PlanExecutionResult, PlanExecutor,
)

__all__ = [
    "ActionLoop", "ActionLoopResult", "ActionOutcome", "ActionReceipt",
    "ActionRequest", "JsonMemory", "MemoryRecord", "MemorySearch",
    "decode_message_sequence", "encode_message_sequence", "DecisionPipeline",
    "CognitiveState", "Evidence", "Hypothesis", "Assessment", "Goal", "Plan", "Observation",
    "EpisodicMemory", "RetrievalHit", "PlanStep", "ExecutablePlan", "OutcomeExperience",
    "ReplanRequest", "PlanExecutionResult", "PlanExecutor",
    "CognitiveSession", "SelectionPolicy", "LearnedEpisodicMemory",
]


def __getattr__(name):
    if name not in {"CognitiveSession", "SelectionPolicy", "LearnedEpisodicMemory"}:
        raise AttributeError(name)
    from importlib import import_module
    value = getattr(import_module(f'{__name__}.cognition'), name)
    globals()[name] = value
    return value
