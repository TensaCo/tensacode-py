from datetime import datetime
from typing import Optional, Literal, Any
from uuid import uuid4
from pydantic import BaseModel, Field, UUID4

from tensacode.core.base.base_engine import BaseEngine


class BaseSession(HasRegistry, BaseModel):
    id: UUID4 = Field(default_factory=uuid4)
    context: BaseModel
    engine: BaseEngine
    created_at: datetime = Field(default_factory=datetime.now)
    state: Literal["open", "closed"] = "open"
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    _registry: ClassVar[Registry["BaseSession"]] = Registry()

    def __enter__(self):
        self.started_at = datetime.now()
        self.state = "open"
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.ended_at = datetime.now()
        self.state = "closed"
