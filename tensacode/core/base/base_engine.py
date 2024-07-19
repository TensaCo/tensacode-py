from contextlib import contextmanager
from typing import Literal
from pydantic import BaseModel

from tensacode.utils.language import Language
from tensacode.internal.protocols.tagged_object import HasID


class BaseEngine(HasID, BaseModel):
    tensacode_version: str
    render_language: Language = "python"
    config: dict[str, Any]

    class Session(HasID, BaseModel):
        pass
