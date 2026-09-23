"""Private non-owning encoder for complete tool models."""
from tensorcode.ops.base import Operation


class SequenceEncoder(Operation):
    """Tokenize text and run an owner's local pretrained sequence encoder.

    Ownership stays with the enclosing model; this callable exposes its encoding
    boundary without registering the shared encoder/decoder weights twice.
    """
    replayable = True

    def __init__(self, model, tokenizer, *, max_tokens=512):
        import weakref
        self._model = weakref.ref(model)
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        import hashlib
        import json
        specification = json.loads(tokenizer.backend_tokenizer.to_str())
        specification["padding"] = None
        specification["truncation"] = None
        self.tokenizer_sha256 = hashlib.sha256(
            json.dumps(specification, sort_keys=True).encode()).hexdigest()

    def forward(self, value, *, context=None):
        if context:
            raise ValueError('SequenceEncoder does not consume context')
        model = self._model()
        if model is None:
            raise RuntimeError('SequenceEncoder owner has been released')
        batch = self.tokenizer(value, padding=True, truncation=True,
                               max_length=self.max_tokens, return_tensors='pt')
        batch = {key: tensor.to(next(model.parameters()).device)
                 for key, tensor in batch.items() if key in ('input_ids', 'attention_mask')}
        encoded = model.get_encoder()(**batch, return_dict=True).last_hidden_state
        return {'encoded': encoded, 'mask': batch['attention_mask']}

    def configuration(self):
        import json
        model = self._model()
        return {'operation': 'tensorcode._internal.vec.sequence.SequenceEncoder',
                'max_tokens': self.max_tokens,
                'model': json.loads(model.config.to_json_string()),
                'tokenizer_sha256': self.tokenizer_sha256,
                'special_tokens': self.tokenizer.special_tokens_map}
