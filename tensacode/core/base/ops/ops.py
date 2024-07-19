from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Dict, Set, Callable, Type, Union
from enum import Enum

from tensacode.core.base.ops.base_op import BaseOp


# Define base classes for each operation type


class CallOp(BaseOp):
    """Call a function by creating or obtaining all necessary args"""


class ChoiceOp(BaseOp):
    """Choose from one of a finite collection of options"""


class CombineOp(BaseOp):
    """Combine attributes from multiple objects"""


class ConvertOp(BaseOp):
    """Convert between different types of objects"""


class CorrectOp(BaseOp):
    """Correct errors in the input"""


class DecideOp(BaseOp):
    """Make a boolean decision"""


class DecodeOp(BaseOp):
    """Decode a representation back into an object"""


class EncodeOp(BaseOp):
    """Encode an object into a representation"""


class ModifyOp(BaseOp):
    """Modify an object"""


class PredictOp(BaseOp):
    """Predict the value of a masked attribute"""


class ProgramOp(BaseOp):
    """Generate functions that can be executed"""


class QueryOp(BaseOp):
    """Query an object for specific information"""


class RetrieveOp(BaseOp):
    """Retrieve an exact value from an objtree"""


class RunOp(BaseOp):
    """Execute functions to produce desired results"""


class SemanticTransferOp(BaseOp):
    """Transfer the semantic meaning of an object"""


class SimilarityOp(BaseOp):
    """Compute the similarity between two objects"""


class SplitOp(BaseOp):
    """Split an object into multiple objects"""


class StoreOp(BaseOp):
    """Store information in an object"""


class StyleTransferOp(BaseOp):
    """Apply the low level style of an object to another object"""


# Base Engine class
class BaseEngine(ABC):
    @abstractmethod
    def execute(self, op: BaseOp, input: Any) -> Any:
        pass


# Engine-specific classes
class NNGraphEngine(BaseEngine):
    def execute(self, op: BaseOp, input: Any) -> Any:
        if not isinstance(op, engine_ops[EngineType.NN_GRAPH][op.__class__.__name__]):
            raise TypeError(f"Invalid operation type for NNGraphEngine: {type(op)}")
        return op.execute(self, input)


class LLMEngine(BaseEngine):
    def execute(self, op: BaseOp, input: Any) -> Any:
        if not isinstance(op, engine_ops[EngineType.LLM][op.__class__.__name__]):
            raise TypeError(f"Invalid operation type for LLMEngine: {type(op)}")
        return op.execute(self, input)


class NNVecEngine(BaseEngine):
    def execute(self, op: BaseOp, input: Any) -> Any:
        if not isinstance(op, engine_ops[EngineType.NN_VEC][op.__class__.__name__]):
            raise TypeError(f"Invalid operation type for NNVecEngine: {type(op)}")
        return op.execute(self, input)
