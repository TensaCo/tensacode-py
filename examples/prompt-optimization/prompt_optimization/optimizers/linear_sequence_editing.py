from pydantic import BaseModel


class Sequence(BaseModel):
    content: str
    trunc_start: int
    trunc_end: int


class LinearSequenceAssembly(BaseModel):
    sequences: list[Sequence]
    