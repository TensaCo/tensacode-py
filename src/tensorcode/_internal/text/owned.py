"""Dependency-light public-operation facade for owned native seq2seq models."""
from collections.abc import Mapping
import json

from tensorcode.ops.base import Operation


class TextObjective(Operation):
    replayable = True

    def __init__(self, owner):
        import weakref
        self._owner = weakref.ref(owner)

    def forward(self, value, *, context=None):
        if not isinstance(value, Mapping) or set(value) != {'inputs', 'targets'}:
            raise ValueError('objective requires inputs and targets')
        inputs = value['inputs']
        if isinstance(inputs, Mapping):
            if set(inputs) != {'value', 'context'} or context:
                raise ValueError('conditioning envelope requires value and context')
            inputs, context = inputs['value'], inputs['context']
        return self._owner().loss(inputs, value['targets'], context=context).clone()

    def parameters(self, recurse=True):
        return self._owner().parameters(recurse=recurse)

    def _operation_identity(self):
        return self._owner()._tool_identity() + '.objective'

    def configuration(self):
        return {'operation': self._operation_identity(), 'model': self._owner().configuration()}


class OwnedTextOperation(Operation):
    semantic_fields = {'instructions'}
    training_inputs_include_targets = True
    artifact_format = 'tensorcode.pretrained'
    artifact_version = 1

    @staticmethod
    def _validated_config(config):
        if not isinstance(config, dict):
            raise ValueError('config must be a JSON object')
        try:
            result = json.loads(json.dumps(config, allow_nan=False))
        except (TypeError, ValueError) as exc:
            raise ValueError('config must contain finite JSON data') from exc
        if result != config:
            raise ValueError('config must use JSON types')
        return result

    def __init__(self, config):
        config = self._validated_config(config)
        allowed = self.semantic_fields | {'native_config', 'tokenizer', 'native_generation_config',
            'native_parameter_aliases', 'generation', 'foundation'}
        if set(config) - allowed:
            raise ValueError(f'Unknown config fields: {sorted(set(config)-allowed)}')
        self._configure_semantics(config)
        from .native import NativeModel
        self.model = NativeModel(config)
        self.config = config
        self._owned = True
        self.training_operation = TextObjective(self)

    def _configure_semantics(self, config):
        self.instructions = config.get('instructions')
        if self.instructions is not None and not isinstance(self.instructions, str):
            raise TypeError('instructions must be a string or None')

    @classmethod
    def from_model(cls, model, **options):
        if set(options) - cls.semantic_fields:
            raise ValueError('Unknown external model options')
        result = cls.__new__(cls)
        result._configure_semantics(options)
        result.model = model
        result._owned = False
        return result

    @classmethod
    def from_foundation(cls, repo, *, revision=None, config=None, **kwargs):
        from tensorcode._internal.vec.text import _load_foundation, _tokenizer_config, _parameter_aliases
        import torch
        settings = cls._validated_config({} if config is None else config)
        if set(settings) - (cls.semantic_fields | {'generation'}):
            raise ValueError('foundation config supports semantic fields and generation only')
        model, tokenizer = _load_foundation(repo, revision, kwargs, decoder=True)
        settings.update(native_config=json.loads(model.config.to_json_string()),
            tokenizer=_tokenizer_config(tokenizer), native_parameter_aliases=_parameter_aliases(model),
            native_generation_config=json.loads(model.generation_config.to_json_string()),
            foundation={'repo': str(repo), 'revision': revision})
        with torch.device('meta'):
            result = cls(settings)
        result.model.model = model
        return result.eval()

    @classmethod
    def _tool_identity(cls):
        return cls.__module__ + '.' + cls.__qualname__

    @property
    def replayable(self):
        return self._owned and not self.model.generation.get('do_sample', False)

    def _require_owned(self):
        if not self._owned:
            raise ValueError('external models do not support owned training or artifacts')

    def configuration(self):
        self._require_owned()
        return self._validated_config({**self.config, **self.model.configuration()})

    def loss(self, value, targets, *, context=None):
        self._require_owned()
        request = self._request(value, context)
        if request.response_schema is not None:
            if not isinstance(targets, Mapping):
                raise ValueError('structured targets must be an explicit JSON mapping')
            from .native import validate_structured
            validate_structured(targets, request.response_schema)
            self._parse(targets)
            targets = json.dumps(dict(targets), sort_keys=True, allow_nan=False)
        elif not isinstance(targets, str):
            raise ValueError('text targets must be a string')
        return self.model.loss(request, targets)

    def operation_bindings(self):
        self._require_owned()
        return {'operation': self, 'objective': self.training_operation}

    def parameters(self, *args, **kwargs):
        self._require_owned()
        return self.model.parameters(*args, **kwargs)

    def named_parameters(self, *args, **kwargs):
        self._require_owned()
        return self.model.named_parameters(*args, **kwargs)

    def named_modules(self, *args, **kwargs):
        self._require_owned()
        return self.model.named_modules(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        self._require_owned()
        return self.model.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        self._require_owned()
        return self.model.load_state_dict(*args, **kwargs)

    def train(self, mode=True):
        self._require_owned()
        self.model.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def to(self, *args, **kwargs):
        self._require_owned()
        self.model.to(*args, **kwargs)
        return self

    def save_pretrained(self, directory):
        self._require_owned()
        from tensorcode._internal.pretrained import PretrainedTool
        return PretrainedTool.save_pretrained(self, directory)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        from tensorcode._internal.pretrained import PretrainedTool
        return PretrainedTool.from_pretrained.__func__(cls, *args, **kwargs)

    @staticmethod
    def _restore_artifact_dtypes(*args):
        from tensorcode._internal.pretrained import PretrainedTool
        return PretrainedTool._restore_artifact_dtypes(*args)

    @classmethod
    def _load_pretrained_config(cls, config, directory):
        return config

    def _save_pretrained_assets(self, directory):
        pass  # Complete tokenizer is embedded in configuration.

    def _default_model_card(self):
        from tensorcode._internal.pretrained import PretrainedTool
        return PretrainedTool._default_model_card(self)
