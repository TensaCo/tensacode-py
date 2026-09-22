"""Ask several structured operations about the same messages."""
from __future__ import annotations

import asyncio
from collections.abc import Mapping
from types import MappingProxyType

from ._structured import InvalidModelOutput, StructuredOperation, require_structured
from .model import ModelOutput


def _questions(questions):
    if not isinstance(questions, Mapping) or not questions:
        raise TypeError("questions must be a nonempty mapping of names to structured operations")
    for name, operation in questions.items():
        if not isinstance(name, str) or not name:
            raise TypeError("question names must be nonempty strings")
        if not isinstance(operation, StructuredOperation):
            raise TypeError("questions must be Classify, Decide, Score or Retrieve operations")
    return dict(questions)


def _shared_model(questions, method):
    """Return the single external model that can answer every question at once."""
    models = {id(operation.model): operation.model for operation in questions.values()}
    if len(models) != 1 or any(operation._owned for operation in questions.values()):
        return None
    (model,) = models.values()
    return model if callable(getattr(model, method, None)) else None


def _results(questions, outputs):
    if not isinstance(outputs, Mapping) or set(outputs) != set(questions):
        raise InvalidModelOutput("Model answers must match the question names")
    if not all(isinstance(output, ModelOutput) for output in outputs.values()):
        raise TypeError("model.complete_questions must return ModelOutput values")
    return MappingProxyType({
        name: operation._parse(require_structured(outputs[name])) for name, operation in questions.items()
    })


def ask(messages, questions, *, context=None):
    """Answer named structured ``questions`` about one message sequence.

    When every question wraps the same external model and that model implements
    ``complete_questions``, all questions travel in one request. Otherwise, and
    whenever a trace is active, each operation is called normally. Returns a
    read-only mapping from question name to that operation's result.
    """
    from tensorcode._internal.tracing import _active

    questions = _questions(questions)
    messages = tuple(messages)
    model = None if _active.get() is not None else _shared_model(questions, "complete_questions")
    if model is None:
        return MappingProxyType({
            name: operation(messages, context=context) for name, operation in questions.items()
        })
    requests = {name: operation._request(messages, context) for name, operation in questions.items()}
    return _results(questions, model.complete_questions(requests))


async def aask(messages, questions, *, context=None):
    """Asynchronous :func:`ask`; uses ``acomplete_questions`` when available."""
    from tensorcode._internal.tracing import _active

    questions = _questions(questions)
    messages = tuple(messages)
    model = None if _active.get() is not None else _shared_model(questions, "acomplete_questions")
    if model is None:
        names = list(questions)
        answers = await asyncio.gather(
            *(questions[name].acall(messages, context=context) for name in names)
        )
        return MappingProxyType(dict(zip(names, answers)))
    requests = {name: operation._request(messages, context) for name, operation in questions.items()}
    return _results(questions, await model.acomplete_questions(requests))
