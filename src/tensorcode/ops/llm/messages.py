from dataclasses import dataclass


@dataclass(frozen=True)
class Message:
    role: str
    content: str

    def __post_init__(self):
        if self.role not in ('system', 'user', 'assistant', 'tool'):
            raise ValueError('Unsupported message role')
        if not isinstance(self.content, str):
            raise TypeError('This text-only message implementation requires a string')
