"""A general agent: vision and language in, a cognitive core, action and language out.

    from tensorcode.agent import Agent
    agent = Agent([some_plugin])
    agent.turn("make a folder called recipes on my desktop").reply

See docs/revival/28-general-agent.md. Experimental, like everything before 1.0.
"""

from .core import Agent, Outcome, Turn, InterpretationDecision, InterpretedMessage, InterpretedImage, InvestigatedInterpretation, InterpretationResolution
from .interpretation import InterpretationWorkspace
from .grounding import MentionBinding, propose_grounding
from .investigation import CandidateHypothesis, InvestigationResult, investigate
from .scene import SceneGraph, SceneProposal, VisualAnchor
from .plugin import Call, Capability, Effect, Informs, Param, Plugin, Precondition
from ..goals import Condition, GoalSpec
from .tasks import Task, TaskLedger
from .filesystem import FileSystemPlugin
from .refinements import RefinementLibrary

__all__ = ["InterpretationResolution", "MentionBinding", "propose_grounding", "InvestigatedInterpretation", "CandidateHypothesis", "InvestigationResult", "investigate", "InterpretedImage", "SceneGraph", "SceneProposal", "VisualAnchor", "InterpretationDecision", "InterpretedMessage", "InterpretationWorkspace", "Agent", "Outcome", "Turn", "Call", "Capability", "Effect", "Informs", "Param", "Plugin", "Precondition", "Condition", "GoalSpec", "Task", "TaskLedger", "FileSystemPlugin", "RefinementLibrary"]
