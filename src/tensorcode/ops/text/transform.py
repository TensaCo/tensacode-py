from ..base import Operation
from .messages import Message
from .model import ModelOutput, ModelRequest
from ._structured import message_sequence, model_configuration


class Transform(Operation):
    def __init__(self, model):
        self.model = model

    def forward(self, value, *, context=None):
        messages = tuple(value)
        if not all(isinstance(m, Message) for m in messages):
            raise TypeError('Expected Message objects')
        # Context roles remain intact; context precedes the primary conversation.
        combined = message_sequence(messages, context)
        complete = getattr(self.model, "complete", None)
        if callable(complete):
            output = complete(ModelRequest(combined))
            if not isinstance(output, ModelOutput):
                raise TypeError("model.complete must return ModelOutput")
            answer = output.text
        else:
            answer = self.model(combined)
        if not isinstance(answer, str):
            raise TypeError('Model must return a string')
        return messages + (Message('assistant', answer),)

    async def aforward(self, value, *, context=None):
        acomplete = getattr(self.model, "acomplete", None)
        if not callable(acomplete):
            return await super().aforward(value, context=context)
        messages = tuple(value)
        combined = message_sequence(messages, context)
        output = await acomplete(ModelRequest(combined))
        if not isinstance(output, ModelOutput) or not isinstance(output.text, str):
            raise TypeError("model.acomplete must return ModelOutput with text")
        return messages + (Message("assistant", output.text),)

    def configuration(self):
        return {"type": "text_transform", "model": model_configuration(self.model)}
