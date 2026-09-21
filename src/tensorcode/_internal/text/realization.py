from tensorcode.ops.base import Operation


class SequenceDecoder(Operation):
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
        return {'operation': 'tensorcode.tools.chatbot.Chatbot.decoder', 'memory': 'explicit-sequence-v1',
                'model': json.loads(model.config.to_json_string()),
                'generation': model.generation_config.to_dict()}
