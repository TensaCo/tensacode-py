from abc import ABC
from typing import Any, Generic
from langchain.chains.base import Chain
import inflect
from tensacode.base.base_engine import BaseEngine
from tensacode.utils.types import T, R


class BaseLLMEngine(Generic[T], BaseEngine[T, str], ABC):
    enc = BaseEngine.enc
    trace = BaseEngine.trace
    DefaultParam = BaseEngine.DefaultParam

    kernel: Chain
    p = inflect.engine()

    def combine(self, *objects: enc[T]) -> str:
        return "\n\n".join(objects)
