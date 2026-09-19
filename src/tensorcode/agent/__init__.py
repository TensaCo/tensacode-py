"""A general agent: vision and language in, a cognitive core, action and language out.

    from tensorcode.agent import Agent
    agent = Agent([some_plugin])
    agent.turn("make a folder called recipes on my desktop").reply

See docs/revival/28-general-agent.md. Experimental, like everything before 1.0.
"""

from .core import Agent, Outcome, Turn, InterpretationDecision, InterpretedMessage, InterpretedImage
from .interpretation import InterpretationWorkspace
from .scene import SceneGraph, SceneProposal, VisualAnchor
from .plugin import Call, Capability, Effect, Informs, Param, Plugin, Precondition
from ..goals import Condition, GoalSpec
from .tasks import Task, TaskLedger
from .filesystem import FileSystemPlugin
from .refinements import RefinementLibrary

__all__ = ["InterpretedImage", "SceneGraph", "SceneProposal", "VisualAnchor", "InterpretationDecision", "InterpretedMessage", "InterpretationWorkspace", "Agent", "Outcome", "Turn", "Call", "Capability", "Effect", "Informs", "Param", "Plugin", "Precondition", "Condition", "GoalSpec", "Task", "TaskLedger", "FileSystemPlugin", "RefinementLibrary"]
