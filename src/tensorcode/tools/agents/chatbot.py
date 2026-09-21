"""A minimal stateful text chatbot using a supplied model."""
from threading import Lock
from ...ops.llm import TextEncoder, TextDecoder, Transform


class Chatbot:
    def __init__(self, *, model=None, encode=None, respond=None, decode=None):
        if (model is None) == (respond is None):
            raise ValueError('Supply exactly one of model or respond')
        self.encode = encode if encode is not None else TextEncoder()
        self.respond = respond if respond is not None else Transform(model)
        self.decode = decode if decode is not None else TextDecoder()
        self._history = ()
        self._lock = Lock()

    @property
    def history(self):
        return self._history

    def __call__(self, value, *, context=None):
        with self._lock:
            messages = self._history + self.encode(value)
            response = self.respond(messages, context=context)
            answer = self.decode(response)
            self._history = response
            return answer
