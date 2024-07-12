from contextlib import contextmanager
from typing import Literal
from pydantic import BaseModel

from tensacode.utils.language import Language


class BaseEngine(BaseModel):
    tensacode_version: str
    render_language: Language = "python"
