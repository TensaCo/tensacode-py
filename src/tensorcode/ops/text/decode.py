from ..base import Operation
from .messages import ImagePart, TextPart


class TextDecoder(Operation):
    replayable = True

    def forward(self, value, *, context=None):
        if context:
            raise ValueError('TextDecoder does not consume context')
        if not value or value[-1].role != 'assistant':
            raise ValueError('Expected a final assistant message')
        content = value[-1].content
        if isinstance(content, str):
            return content
        if any(isinstance(part, ImagePart) for part in content):
            raise ValueError("Assistant message contains an image and cannot be decoded as text")
        return "".join(part.text for part in content if isinstance(part, TextPart))


class Decode(Operation):
    """Owned local seq2seq language realization from explicit memory tensors.

    The owner registers the model parameters. This operation keeps a weak
    reference so a shared encoder/decoder model is saved exactly once.
    """
    replayable = True

    def __init__(self, model):
        import weakref
        self._model = weakref.ref(model)

    def forward(self, value, *, context=None):
        from transformers.modeling_outputs import BaseModelOutput
        context = dict(context or {})
        model = self._model()
        if model is None:
            raise RuntimeError('Decode owner has been released')
        memory = BaseModelOutput(last_hidden_state=value['conditioning'])
        arguments = dict(encoder_outputs=memory, attention_mask=value['mask'])
        if 'labels' in value:
            result = model(**arguments, labels=value['labels'], return_dict=True, use_cache=False)
            return {'loss': result.loss, 'logits': result.logits}
        return model.generate(**arguments, **context)

    def configuration(self):
        import json
        model = self._model()
        return {'operation': 'tensorcode.ops.text.Decode', 'memory': 'explicit-sequence-v1',
                'model': json.loads(model.config.to_json_string()),
                'generation': model.generation_config.to_dict()}
