from ..._internal.text.owned import OwnedTextOperation
from .messages import Message
from .model import ModelOutput, ModelRequest
from ._structured import message_sequence, model_configuration


class Transform(OwnedTextOperation):
    """Generate an assistant reply using an owned native seq2seq model.

    ``from_model`` explicitly wraps an external provider without owned artifacts.
    """
    def _request(self, value, context):
        return ModelRequest(message_sequence(value, context), instructions=self.instructions)

    def forward(self, value, *, context=None):
        messages = tuple(value)
        if not all(isinstance(m, Message) for m in messages):
            raise TypeError('Expected Message objects')
        # Context roles remain intact; context precedes the primary conversation.
        combined = message_sequence(messages, context)
        complete = getattr(self.model, "complete", None)
        if callable(complete):
            output = complete(self._request(messages, context))
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
        output = await acomplete(self._request(messages, context))
        if not isinstance(output, ModelOutput) or not isinstance(output.text, str):
            raise TypeError("model.acomplete must return ModelOutput with text")
        return messages + (Message("assistant", output.text),)

    def configuration(self):
        if self._owned:
            return super().configuration()
        return {"type": "text_transform", "instructions": self.instructions,
                "model": model_configuration(self.model)}
