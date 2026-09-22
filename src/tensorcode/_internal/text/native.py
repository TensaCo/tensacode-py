"""Native local sequence model; targets only enter teacher-forced labels."""
import json
import threading
from collections.abc import Mapping
import torch
from torch import nn
from tensorcode._internal.vec.text import _native_config, _tokenizer, _tokenizer_config, _restore_parameter_aliases, _parameter_aliases
from tensorcode.ops.text.messages import TextPart
from tensorcode.ops.text.model import ModelOutput
from tensorcode.ops.text._structured import InvalidModelOutput


def validate_structured(value, schema):
    if not isinstance(value, Mapping) or set(value) != set(schema['required']):
        raise InvalidModelOutput('Structured output must contain exactly the required fields')
    # Semantic parsers validate values, distributions, and configured choices.


class NativeModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._execution_lock = threading.RLock()
        from transformers import AutoModelForSeq2SeqLM, GenerationConfig
        if not {'native_config', 'tokenizer'} <= set(config):
            raise ValueError('config requires native_config and tokenizer')
        native = _native_config(config['native_config'])
        if not native.is_encoder_decoder:
            raise ValueError('owned text operations require a native seq2seq architecture')
        self.model = AutoModelForSeq2SeqLM.from_config(native)
        _restore_parameter_aliases(self.model, config.get('native_parameter_aliases'))
        if 'native_generation_config' in config:
            self.model.generation_config = GenerationConfig.from_dict(config['native_generation_config'])
        self.tokenizer = _tokenizer(config['tokenizer'])
        self.generation = {'max_new_tokens': 32, 'do_sample': False, **config.get('generation', {})}
        allowed = {'max_new_tokens','min_new_tokens','num_beams','do_sample','temperature','top_k','top_p','repetition_penalty','length_penalty','early_stopping'}
        if set(self.generation) - allowed:
            raise ValueError('unsupported generation setting')

    def configuration(self):
        return {'native_config': json.loads(self.model.config.to_json_string()),
            'native_parameter_aliases': _parameter_aliases(self.model),
            'tokenizer': _tokenizer_config(self.tokenizer), 'generation': dict(self.generation),
            'native_generation_config': json.loads(self.model.generation_config.to_json_string())}

    def prompt(self, request):
        messages = []
        for message in request.messages:
            content = message.content
            if not isinstance(content, str):
                if not all(isinstance(part, TextPart) for part in content):
                    raise ValueError('native text models only support text message content')
                content = ''.join(part.text for part in content)
            messages.append({'role': message.role, 'content': content})
        return json.dumps({'messages': messages, 'instructions': request.instructions,
            'response_schema': request.response_schema, 'schema_name': request.schema_name}, sort_keys=True)

    def inputs(self, request):
        tokens = self.tokenizer([self.prompt(request)], padding=True, return_tensors='pt')
        device = next(self.model.parameters()).device
        return {key: value.to(device) for key, value in tokens.items() if key in ('input_ids','attention_mask')}

    def train(self, mode=True):
        with self._execution_lock:
            return super().train(mode)

    def complete(self, request):
        # Async operation fallbacks share native mode flags and tokenizer state.
        # Restore modes before another generation or teacher-forced call starts.
        with self._execution_lock:
            return self._complete(request)

    def _complete(self, request):
        modes = {module: module.training for module in self.model.modules()}
        try:
            self.model.eval()
            ids = self.model.generate(**self.inputs(request), **self.generation)
        finally:
            for module, mode in modes.items():
                module.training = mode
        text = self.tokenizer.decode(ids[0], skip_special_tokens=True)
        if request.response_schema is None:
            return ModelOutput(text=text)
        try:
            value = json.loads(text)
        except (TypeError, ValueError) as exc:
            raise InvalidModelOutput('Native model did not generate valid JSON') from exc
        validate_structured(value, request.response_schema)
        return ModelOutput(text=text, structured=value)

    def loss(self, request, target):
        with self._execution_lock:
            return self._loss(request, target)

    def alternative_log_likelihoods(self, request, alternatives, *, normalization='sum'):
        """Return one log-likelihood per ``(display, target)`` alternative.

        The prompt is encoded once; every target is scored by the decoder in a
        single batch. Gradients follow the current autograd/module mode.
        """
        with self._execution_lock:
            return self._alternative_log_likelihoods(request, alternatives, normalization)

    def _alternative_log_likelihoods(self, request, alternatives, normalization):
        from transformers.modeling_outputs import BaseModelOutput
        if normalization not in ('sum', 'mean'):
            raise ValueError('normalization must be sum or mean')
        tokens = self.tokenizer([alternative_prompt(request, alternatives)], return_tensors='pt')
        device = next(self.model.parameters()).device
        input_ids = tokens['input_ids'].to(device)
        attention = tokens['attention_mask'].to(device)
        encoded = self.model.get_encoder()(input_ids=input_ids, attention_mask=attention)
        count = len(alternatives)
        targets = self.tokenizer([target for _display, target in alternatives], padding=True, return_tensors='pt')
        labels = targets['input_ids'].to(device)
        valid = targets['attention_mask'].to(device).bool()
        rows = {tuple(row[mask].tolist()) for row, mask in zip(labels, valid)}
        if len(rows) != count:
            raise ValueError('likelihood alternatives must tokenize to distinct sequences')
        labels = labels.masked_fill(~valid, -100)
        hidden = encoded.last_hidden_state.expand(count, -1, -1)
        logits = self.model(encoder_outputs=BaseModelOutput(last_hidden_state=hidden),
            attention_mask=attention.expand(count, -1), labels=labels).logits
        token_scores = logits.log_softmax(-1).gather(-1, labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        scores = (token_scores * valid).sum(-1)
        if normalization == 'mean':
            scores = scores / valid.sum(-1)
        return scores

    def score_alternatives(self, request, alternatives, *, normalization='sum'):
        """Evaluation-mode, gradient-free alternative scores as Python floats."""
        import torch
        with self._execution_lock:
            modes = {module: module.training for module in self.model.modules()}
            try:
                self.model.eval()
                with torch.no_grad():
                    scores = self._alternative_log_likelihoods(request, alternatives, normalization)
            finally:
                for module, mode in modes.items():
                    module.training = mode
        return [float(value) for value in scores.float().cpu()]


    def _loss(self, request, target):
        inputs = self.inputs(request)
        tokens = self.tokenizer([target], padding=True, return_tensors='pt')
        labels = tokens['input_ids'].to(inputs['input_ids'].device)
        labels = labels.masked_fill(~tokens['attention_mask'].to(labels.device).bool(), -100)
        return self.model(**inputs, labels=labels).loss


def alternative_prompt(request, alternatives):
    """Plain-text serialization shared by likelihood scoring and its training."""
    lines = [request.instructions] if request.instructions else []
    for message in request.messages:
        content = message.content
        if not isinstance(content, str):
            if not all(isinstance(part, TextPart) for part in content):
                raise ValueError('native text models only support text message content')
            content = ''.join(part.text for part in content)
        lines.append(f'{message.role}: {content}')
    lines.append('Options:')
    lines.extend(f'- {display}' for display, _target in alternatives)
    lines.append('Answer:')
    return '\n'.join(lines)
