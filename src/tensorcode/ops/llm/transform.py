from ..base import Operation
from .messages import Message


class Transform(Operation):
    def __init__(self, model):
        self.model = model

    def forward(self, value, *, context=None):
        messages = tuple(value)
        if not all(isinstance(m, Message) for m in messages):
            raise TypeError('Expected Message objects')
        # Context roles remain intact; context precedes the primary conversation.
        conditioning = tuple(m for group in (context or {}).values() for m in group)
        if not all(isinstance(m, Message) for m in conditioning):
            raise TypeError('Context must contain message sequences')
        answer = self.model(conditioning + messages)
        if not isinstance(answer, str):
            raise TypeError('Model must return a string')
        return messages + (Message('assistant', answer),)
