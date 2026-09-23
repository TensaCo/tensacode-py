"""Data-configured tensor readouts and native transformer latent bridges."""
from __future__ import annotations

import json
from collections.abc import Mapping
import torch
from torch import nn
from tensorcode._internal.latent_ops import LatentOperation, as_sequence
from tensorcode.ops.vec.latent import Latent, Space
from tensorcode.ops.base import Operation


class Objective(Operation):
    replayable = True

    def __init__(self, owner):
        import weakref
        self._owner = weakref.ref(owner)

    def _operation_identity(self):
        return self._owner()._tool_identity() + '.objective'

    def parameters(self, recurse=True):
        return self._owner().parameters(recurse=recurse)

    def configuration(self):
        return {'operation': self._operation_identity(), 'model': self._owner().configuration()}

    def forward(self, value, *, context=None):
        if not isinstance(value, Mapping) or set(value) != {'inputs', 'targets'}:
            raise ValueError('objective envelope requires inputs and targets')
        inputs=value['inputs']
        if isinstance(inputs,Mapping):
            if set(inputs) != {'value','context'} or context:
                raise ValueError('conditioning envelope requires exactly value and context')
            context=inputs['context'];inputs=inputs['value']
        return self._owner().loss(inputs,value['targets'],context=context).clone()


def positive(value, name):
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return value


def network(input_dim, output_dim, architecture, config):
    hidden = config.get('hidden_dimensions', [])
    if not isinstance(hidden, list):
        raise ValueError('hidden_dimensions must be a list')
    if architecture == 'linear' and hidden:
        raise ValueError('linear architecture does not use hidden_dimensions')
    if architecture == 'mlp' and not hidden:
        raise ValueError('mlp requires nonempty hidden_dimensions')
    widths = [input_dim, *[positive(v, 'hidden dimension') for v in hidden], output_dim]
    layers = []
    for index, (left, right) in enumerate(zip(widths, widths[1:])):
        layers.append(nn.Linear(left, right))
        if index < len(widths)-2:
            layers.append(nn.GELU())
    return layers[0] if len(layers) == 1 else nn.Sequential(*layers)


class OwnedMap(LatentOperation):
    """Shared architecture; public subclasses define their result contracts."""
    kind = 'transform'
    training_inputs_include_targets = True

    def __init__(self, config):
        super().__init__(config)
        config = self.config
        common = {'architecture','input_space','hidden_dimensions','native_config','readout','foundation'}
        extra = {'transform':{'output_space'}, 'classify':{'labels'}, 'decode':{'output_dimensions','output'}}[self.kind]
        if set(config) - common - extra:
            raise ValueError(f'unknown configuration fields: {sorted(set(config)-common-extra)}')
        self.input_space = Space(**config['input_space'])
        self.config['input_space'] = self.input_space.configuration()
        if self.kind == 'transform':
            self.output_space = Space(**config['output_space'])
            self.config['output_space'] = self.output_space.configuration()
            width = self.output_space.dimensions
        elif self.kind == 'classify':
            labels = config['labels']
            if not isinstance(labels,list) or not labels or not all(isinstance(v,str) and v for v in labels) or len(set(labels)) != len(labels):
                raise ValueError('labels must be nonempty unique strings')
            self.labels = tuple(labels)
            width = len(labels)
        else:
            width = positive(config['output_dimensions'], 'output_dimensions')
            self.output = config['output']
            if not isinstance(self.output,str) or not self.output.strip():
                raise ValueError('output must be a nonempty description')
        architecture = config.get('architecture', 'linear')
        self.config['architecture'] = architecture
        self.readout = config.get('readout','sequence' if self.kind == 'transform' else 'pooled')
        if self.readout not in ('sequence','pooled'):
            raise ValueError('readout must be sequence or pooled')
        if self.kind == 'classify' and self.readout != 'pooled':
            raise ValueError('classification requires pooled readout')
        self.config['readout'] = self.readout
        if self.kind == 'transform':
            organization = self.input_space.organization if self.readout == 'sequence' else 'feature'
            if self.output_space.organization != organization:
                raise ValueError('output_space organization must match readout')
        self._supplied = False
        if architecture in ('linear','mlp'):
            if 'native_config' in config or 'foundation' in config:
                raise ValueError('native_config and foundation require transformer architecture')
            self.module = network(self.input_space.dimensions,width,architecture,config)
        elif architecture == 'transformer':
            if 'hidden_dimensions' in config:
                raise ValueError('transformer uses native_config, not hidden_dimensions')
            from transformers import AutoModel
            from .text import _native_config, _width
            native = _native_config(config['native_config'])
            # Encoder-only models expose stable inputs_embeds/last_hidden_state.
            if native.model_type not in ('bert','roberta','distilbert'):
                raise ValueError('supported native transformer architectures: bert, roberta, distilbert')
            self.model = AutoModel.from_config(native)
            self.config['native_config'] = json.loads(self.model.config.to_json_string())
            self.input_projection = nn.Linear(self.input_space.dimensions, _width(native))
            self.module = nn.Linear(_width(native),width)
        else:
            raise ValueError('architecture must be linear, mlp or transformer')
        self.objective = Objective(self)

    @classmethod
    def from_module(cls, module, **kwargs):
        """Advanced: wrap a supplied ``torch.nn.Module``; it cannot ``save_pretrained``."""
        from .adapter import TensorAdapter
        instance = cls.__new__(cls)
        nn.Module.__init__(instance)
        adapter_kwargs = {k:kwargs.pop(k) for k in ('combine','input_space','output_space') if k in kwargs}
        adapter = TensorAdapter(module,**adapter_kwargs)
        instance.module = adapter.module
        instance.combine = adapter.combine
        instance.input_space = adapter.input_space
        instance.output_space = adapter.output_space
        if cls.kind == 'classify':
            labels=kwargs.pop('labels')
            if not labels or not all(isinstance(v,str) and v for v in labels) or len(set(labels)) != len(labels):
                raise ValueError('labels must be nonempty and unique')
            instance.labels=tuple(labels)
        if cls.kind == 'decode':
            instance.output=kwargs.pop('output')
            if not isinstance(instance.output,str) or not instance.output.strip():
                raise ValueError('output description must be nonempty')
        if kwargs:
            raise TypeError(f'unsupported from_module arguments: {sorted(kwargs)}')
        instance._supplied = True
        instance.objective = Objective(instance)
        return instance

    @classmethod
    def from_foundation(cls, repo, *, input_space, revision=None, **kwargs):
        """Load a bert/roberta/distilbert backbone; input bridge and head start untrained."""
        from transformers import AutoModel
        config_keys={'output_space','labels','output_dimensions','output','readout'}
        config={k:kwargs.pop(k) for k in tuple(kwargs) if k in config_keys}
        if kwargs.pop('trust_remote_code',False) or kwargs.pop('use_safetensors',True) is not True:
            raise ValueError('foundation requires native code and safetensors')
        model, info = AutoModel.from_pretrained(repo,revision=revision,trust_remote_code=False,use_safetensors=True,output_loading_info=True,**kwargs)
        if any(info.get(k) for k in ('missing_keys','mismatched_keys','error_msgs')):
            raise ValueError(f'foundation has missing or incompatible weights: {info}')
        for key in ('output_space',):
            if isinstance(config.get(key),Space):
                config[key] = config[key].configuration()
        config.update(architecture='transformer',input_space=input_space.configuration() if isinstance(input_space,Space) else input_space,
                      native_config=json.loads(model.config.to_json_string()),foundation={'repo':str(repo),'revision':revision,'input_bridge':'untrained','output_head':'untrained'})
        # Build only architecture metadata for the soon-to-be-replaced backbone.
        # Materialize the newly authored bridges without touching loaded weights.
        with torch.device('meta'):
            instance=cls(config)
        instance.model=model
        parameter=next(model.parameters())
        for name in ('input_projection','module'):
            layer=getattr(instance,name)
            setattr(instance,name,nn.Linear(layer.in_features,layer.out_features,
                bias=layer.bias is not None,device=parameter.device,dtype=parameter.dtype))
        return instance.eval()

    def configuration(self):
        """JSON configuration that reconstructs this operation."""
        if self._supplied:
            from .adapter import TensorAdapter
            config=TensorAdapter.configuration(self)
            if self.kind == 'classify': config['labels']=list(self.labels)
            if self.kind == 'decode': config['output']=self.output
            return config
        return super().configuration()

    @property
    def training_operation(self):
        """Objective operation used for supervised training."""
        return self.objective

    def operation_bindings(self):
        """Named operations for tracing, experience and checkpoints."""
        return {**super().operation_bindings(), 'objective': self.training_operation}

    def save_pretrained(self, directory):
        """Save configuration and weights; rejected for ``from_module`` operations."""
        if self._supplied:
            raise ValueError('supplied modules have no declarative reconstruction; cannot save_pretrained')
        return super().save_pretrained(directory)

    def _tensor(self,value,context):
        if self._supplied:
            from .adapter import TensorAdapter
            return TensorAdapter.forward(self,value,context=context)
        if context is None: context={}
        if not isinstance(context,Mapping) or set(context)-{'latents'}:
            raise ValueError('context supports only latents')
        prefixes=context.get('latents',[])
        if not isinstance(prefixes,(tuple,list)):
            raise ValueError('context latents must be an ordered list')
        x,mask=as_sequence(value,self.input_space)
        original_shape=value.tensor.shape
        if self.config['architecture'] == 'transformer':
            pairs=[as_sequence(v,self.input_space) for v in prefixes]
            if any(t.shape[0] != x.shape[0] or t.device != x.device or t.dtype != x.dtype for t,m in pairs):
                raise ValueError('context latents must have matching batch, device and dtype')
            count=sum(t.shape[1] for t,m in pairs)
            inputs=torch.cat([*[t for t,m in pairs],x],dim=1)
            attention=torch.cat([*[m for t,m in pairs],mask],dim=1)
            hidden=self.model(inputs_embeds=self.input_projection(inputs),attention_mask=attention,return_dict=True).last_hidden_state[:,count:]
        else:
            if prefixes: raise ValueError('latent context requires transformer architecture')
            hidden=x
        if self.readout == 'pooled':
            hidden=hidden.masked_fill(~mask[...,None],0).sum(1)/mask.sum(1,keepdim=True)
            result=self.module(hidden)
            single=(self.input_space.organization=='feature' and value.tensor.ndim==1) or (self.input_space.organization=='sequence' and value.tensor.ndim==2)
            return result[0] if single else result
        result=self.module(hidden).masked_fill(~mask[...,None],0)
        return result.reshape(*original_shape[:-1],result.shape[-1])

    def forward(self,value,*,context=None):
        """Map a ``Latent`` (optionally with ``context={'latents': [...]}``)."""
        result=self._tensor(value,context)
        if self._supplied: return result
        if self.kind == 'transform':
            return value.with_tensor(result,space=self.output_space,mask=value.mask if self.readout=='sequence' else None,coordinates=value.coordinates if self.readout=='sequence' else None)
        return result

    def loss(self,value,targets,*,context=None):
        """Cross-entropy for labels, otherwise masked MSE against target tensors."""
        result=self.forward(value,context=context)
        if self.kind == 'classify':
            logits=result.logits
            if isinstance(targets,str): targets=[targets]
            if isinstance(targets,(tuple,list)) and all(isinstance(t,str) for t in targets):
                targets=torch.tensor([self.labels.index(t) for t in targets],device=logits.device)
                if logits.ndim==1: targets=targets.squeeze(0)
            if not isinstance(targets,torch.Tensor) or targets.shape != logits.shape[:-1] or targets.dtype != torch.long:
                raise ValueError('classification targets must be long indices matching the batch')
            return nn.functional.cross_entropy(logits,targets.to(logits.device))
        tensor=result.tensor if isinstance(result,Latent) else result
        if isinstance(targets,Latent):
            if not isinstance(result,Latent) or targets.space != result.space:
                raise ValueError('target space must match output space')
            target=targets.tensor
        else: target=targets
        if not isinstance(target,torch.Tensor) or target.shape != tensor.shape:
            raise ValueError('targets must match output shape')
        mask=result.mask if isinstance(result,Latent) else (value.mask if not self._supplied and self.readout=='sequence' else None)
        if isinstance(targets,Latent) and targets.mask is not None:
            if targets.mask.dtype != torch.bool: raise ValueError('target mask must be boolean')
            mask=targets.mask if mask is None else mask & targets.mask
        target=target.to(tensor)
        if mask is not None:
            if not mask.any(): raise ValueError('loss requires valid target positions')
            tensor=tensor[mask];target=target[mask]
        if not torch.isfinite(target).all(): raise ValueError('valid targets must be finite')
        return (tensor-target).square().mean()
