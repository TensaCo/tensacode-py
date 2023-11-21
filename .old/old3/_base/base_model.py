from __future__ import annotations
from abc import abstractmethod
from enum import Enum, auto

from old3._base.ledger import Ledger


class BaseModel:

    # class attrs

    __CURRENT_MODEL_STACK: list[BaseModel] = []

    @staticmethod
    def _CURRENT_MODEL(cls) -> BaseModel:
        if len(BaseModel.__CURRENT_MODEL_STACK) == 0:
            raise RuntimeError("No model is currently active.")
        return BaseModel.__CURRENT_MODEL_STACK[-1]

    # instance attrs

    ledger: Ledger
    reward_bucket: float = 0

    # methods

    ## default model management

    def __enter__(self):
        BaseModel.__CURRENT_MODEL_STACK.append(self)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.__CURRENT_MODEL_STACK[-1] is not self:
            raise RuntimeError("Unbalanced model context __enter__ and __exit__ calls.")
        self.__CURRENT_MODEL_STACK.pop()

    class LearningMode(Enum):
        ignore = auto()  # model ignores all feedback
        manually = auto()  # usser must invoke .learn() manually
        on_feedback = auto()  # model invokes .learn() when feedback is received
        autonomously = auto()  # model invokes .learn() when it feels like it

    @abstractmethod
    def reward(self, reward):
        ...

    def add_loss(self, loss):
        self.reward(-loss)
