"""Owned native text transformers with explicit latent-space bridges.

Encoder and decoder context is an ordered list of latent prefixes; targets
enter only ``loss``.
Configuration embeds the complete fast tokenizer and native architecture.
"""
from __future__ import annotations

from collections.abc import Mapping
import json

import torch
from torch import nn

from tensorcode._internal.latent_ops import LatentOperation, as_sequence
from tensorcode.ops.vec.latent import Latent, Space
from tensorcode.ops.base import Operation


def _native_config(config):
    from transformers import AutoConfig
    data = dict(config)
    model_type = data.pop('model_type')
    native=AutoConfig.for_model(model_type, **data)
    # T5Config 5.17 rewrites the legacy tie flag while deriving output scaling.
    # Artifacts must retain both authored topology and numerical behavior.
    for name in ('tie_word_embeddings','scale_decoder_outputs'):
        if name in data:
            setattr(native,name,data[name])
    return native


def _tokenizer_config(tokenizer):
    if not tokenizer.is_fast:
        raise ValueError('a fast tokenizer is required for complete offline artifacts')
    backend = json.loads(tokenizer.backend_tokenizer.to_str())
    # Fast tokenizers rewrite these execution settings on every batch call.
    # Wrapper options below own their persistent semantics.
    backend['padding'] = None
    backend['truncation'] = None
    return {'json': json.dumps(backend,sort_keys=True,separators=(',',':')),
            'options': {name:getattr(tokenizer,name) for name in (
                'clean_up_tokenization_spaces','model_max_length','model_input_names','split_special_tokens')},
            'special_tokens': {key: str(value) if not isinstance(value,list) else [str(v) for v in value]
                               for key,value in tokenizer.special_tokens_map.items()},
            'padding_side': tokenizer.padding_side, 'truncation_side': tokenizer.truncation_side}


def _tokenizer(config):
    from tokenizers import Tokenizer
    from transformers import PreTrainedTokenizerFast
    return PreTrainedTokenizerFast(tokenizer_object=Tokenizer.from_str(config['json']),
                                  **config['special_tokens'],**config.get('options',{}),padding_side=config.get('padding_side','right'),
                                  truncation_side=config.get('truncation_side','right'))


def _load_foundation(repo, revision, kwargs, *, decoder=False):
    from transformers import AutoConfig, AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer
    kwargs = dict(kwargs)
    if kwargs.pop('trust_remote_code', False) or kwargs.pop('use_safetensors', True) is not True:
        raise ValueError('foundation loading requires native code and safetensors')
    native = AutoConfig.from_pretrained(repo,revision=revision,trust_remote_code=False,**kwargs)
    cls = AutoModelForSeq2SeqLM if decoder or native.is_encoder_decoder else AutoModel
    model, info = cls.from_pretrained(repo,revision=revision,trust_remote_code=False,use_safetensors=True,output_loading_info=True,**kwargs)
    if info.get('missing_keys') or info.get('mismatched_keys') or info.get('error_msgs'):
        raise ValueError(f'foundation has missing or incompatible weights: {info}')
    raw_config, _ = type(native).get_config_dict(repo,revision=revision,**kwargs)
    for name in ('tie_word_embeddings','scale_decoder_outputs'):
        if name in raw_config:
            setattr(model.config,name,raw_config[name])
    tokenizer = AutoTokenizer.from_pretrained(repo,revision=revision,trust_remote_code=False,use_fast=True,**kwargs)
    return model,tokenizer


def _parameter_aliases(model):
    canonical={}
    aliases={}
    for name,parameter in model.named_parameters(remove_duplicate=False):
        aliases[name]=canonical.setdefault(id(parameter),name)
    return aliases


def _restore_parameter_aliases(model, aliases):
    """Restore actual native parameter sharing, including partially untied T5."""
    if aliases is None:
        return
    params=dict(model.named_parameters(remove_duplicate=False))
    if set(params)!=set(aliases):
        raise ValueError('native parameter topology differs from configuration')
    seen={}
    for name,source in aliases.items():
        if source not in params or aliases[source]!=source:
            raise ValueError('invalid native parameter alias topology')
        if name==source:
            parameter=params[name]
            if id(parameter) in seen:
                parameter=nn.Parameter(parameter.detach().clone(),requires_grad=parameter.requires_grad)
            seen[id(parameter)]=name
            params[name]=parameter
    for name,source in aliases.items():
        parent,_,attribute=name.rpartition('.')
        setattr(model.get_submodule(parent),attribute,params[source])


def _width(config):
    return getattr(config,'hidden_size',None) or config.d_model


def _context(context, key):
    if context is None:
        return []
    if not isinstance(context,Mapping) or set(context)-{key}:
        raise ValueError(f'context supports only {key!r}')
    values = context.get(key,[])
    if not isinstance(values,(list,tuple)):
        raise ValueError(f'context[{key!r}] must be an ordered list')
    return values


class TextEncoder(LatentOperation):
    """Native states, masked mean, or an owned appended OUTPUT_ENCODING token.

    The new token starts untrained even with inherited foundation weights.
    Matching widths do not imply a shared semantic space.
    """
    replayable = True

    def __init__(self, config):
        super().__init__(config)
        allowed = {'native_config', 'native_parameter_aliases', 'tokenizer', 'readout',
                   'output_space', 'context_space', 'foundation'}
        if set(config) - allowed:
            raise ValueError(f'Unknown configuration fields: {sorted(set(config) - allowed)}; use output_space and readout')
        from transformers import AutoModel, AutoModelForSeq2SeqLM
        native = _native_config(config['native_config'])
        factory = AutoModelForSeq2SeqLM if native.is_encoder_decoder else AutoModel
        self.model = factory.from_config(native)
        _restore_parameter_aliases(self.model,config.get("native_parameter_aliases"))
        self.tokenizer = _tokenizer(config['tokenizer'])
        self.readout = config.get('readout','sequence')
        if self.readout not in ('sequence','pooled','output_encoding'):
            raise ValueError('readout must be sequence, pooled or output_encoding')
        if self.readout == 'output_encoding':
            self.output_encoding = nn.Parameter(torch.empty(1, 1, self.model.get_input_embeddings().weight.shape[-1]))
            nn.init.normal_(self.output_encoding, std=0.02)
        self.output_space = Space(**config['output_space'])
        if self.output_space.dimensions != _width(native) or self.output_space.organization != ('sequence' if self.readout=='sequence' else 'feature'):
            raise ValueError('output space must match native width and readout organization')

        context_config = config.get('context_space')
        self.context_space = Space(**context_config) if context_config else None
        encoder = self.model.get_encoder() if native.is_encoder_decoder else self.model
        if self.context_space and self.context_space.dimensions != encoder.get_input_embeddings().weight.shape[-1]:
            raise ValueError('context_space must match native input embedding width')

    @classmethod
    def from_foundation(cls, repo, *, revision=None, readout='sequence', output_space=None, context_space=None, **kwargs):
        model,tokenizer = _load_foundation(repo,revision,kwargs)
        space = output_space or Space(f'{repo}:encoder:{readout}',_width(model.config),version=revision or 'unversioned',organization='sequence' if readout=='sequence' else 'feature')
        config = {'native_config':json.loads(model.config.to_json_string()),'native_parameter_aliases':_parameter_aliases(model),'tokenizer':_tokenizer_config(tokenizer),
                  'readout':readout,'context_space':context_space.configuration() if isinstance(context_space,Space) else context_space,'output_space':space.configuration(),
                  'foundation':{'repo':str(repo),'revision':revision}}
        with torch.device("meta"):
            result = cls(config)
        result.model = model
        if readout == 'output_encoding':
            parameter = model.get_input_embeddings().weight
            result.output_encoding = nn.Parameter(torch.empty(1, 1, parameter.shape[-1],
                device=parameter.device, dtype=parameter.dtype))
            nn.init.normal_(result.output_encoding, std=0.02)
        return result.eval()

    def configuration(self):
        config=super().configuration()
        config['tokenizer']=_tokenizer_config(self.tokenizer)
        return self._validated_config(config)

    def forward(self, value, *, context=None):
        single = isinstance(value,str)
        texts = [value] if single else value
        if not isinstance(texts,(list,tuple)) or not texts or not all(isinstance(t,str) for t in texts):
            raise ValueError('text input must be a string or nonempty list of strings')
        prefixes = _context(context,'latents')
        if prefixes and self.context_space is None:
            raise ValueError('context requires an explicit context_space')
        tokens = self.tokenizer(texts,padding=True,return_tensors='pt')
        device = next(self.model.parameters()).device
        encoder = self.model.get_encoder() if self.model.config.is_encoder_decoder else self.model
        inputs = {k:v.to(device) for k,v in tokens.items() if k in ('input_ids','attention_mask')}
        mask = inputs['attention_mask'].bool()
        sources = []
        if self.readout == 'output_encoding':
            embeds = encoder.get_input_embeddings()(inputs['input_ids'])
            pieces, masks = [], []
            for prefix in prefixes:
                sequence, prefix_mask = as_sequence(prefix,self.context_space)
                if sequence.shape[0] != embeds.shape[0]:
                    raise ValueError('context batch must match text batch')
                pieces.append(sequence.to(device=embeds.device,dtype=embeds.dtype))
                masks.append(prefix_mask.to(embeds.device))
                sources.extend(prefix.sources)
            combined = torch.cat([*pieces, embeds], 1)
            valid = torch.cat([*masks, mask], 1)
            # Absolute readout positions must not depend on batch padding.
            rows = [torch.cat([row[keep], self.output_encoding[0]], 0)
                    for row, keep in zip(combined, valid)]
            lengths = valid.sum(1) + 1
            limit = getattr(encoder.config, 'max_position_embeddings', None)
            positions = getattr(getattr(encoder, 'embeddings', None), 'position_embeddings', None)
            # RoBERTa-family embedding positions begin after the reserved pad
            # position. BERT/ALBERT position embeddings have no padding index.
            if positions is not None and positions.padding_idx is not None and limit is not None:
                limit -= positions.padding_idx + 1
            if limit is not None and lengths.max().item() > limit:
                raise ValueError('input, context and OUTPUT_ENCODING exceed native position capacity')
            packed = nn.utils.rnn.pad_sequence(rows, batch_first=True)
            packed_mask = torch.arange(packed.shape[1], device=device)[None, :] < lengths[:, None]
            hidden = encoder(inputs_embeds=packed, attention_mask=packed_mask).last_hidden_state
            states = hidden[torch.arange(hidden.shape[0], device=device), lengths - 1]
            mask = torch.ones(states.shape[0], device=device, dtype=torch.bool)
        elif prefixes:
            embeds = encoder.get_input_embeddings()(inputs['input_ids'])
            pieces, masks = [], []
            for prefix in prefixes:
                sequence, prefix_mask = as_sequence(prefix,self.context_space)
                if sequence.shape[0] != embeds.shape[0]:
                    raise ValueError('context batch must match text batch')
                pieces.append(sequence.to(device=embeds.device,dtype=embeds.dtype))
                masks.append(prefix_mask.to(embeds.device))
                sources.extend(prefix.sources)
            combined = torch.cat([*pieces, embeds], 1)
            valid = torch.cat([*masks, mask], 1)
            lengths = valid.sum(1)
            if not lengths.all():
                raise ValueError('every input sequence must contain an unmasked position')
            packed = nn.utils.rnn.pad_sequence(
                [row[keep] for row, keep in zip(combined, valid)], batch_first=True)
            packed_mask = torch.arange(packed.shape[1], device=device)[None, :] < lengths[:, None]
            hidden = encoder(inputs_embeds=packed, attention_mask=packed_mask).last_hidden_state
            # Restore the primary text layout, retaining its original mask.
            prefix_lengths = torch.cat(masks, 1).sum(1)
            positions = prefix_lengths[:, None] + mask.long().cumsum(1) - 1
            positions = positions.masked_fill(~mask, 0)
            states = hidden.gather(1, positions.unsqueeze(-1).expand(-1, -1, hidden.shape[-1]))
            states = states.masked_fill(~mask.unsqueeze(-1), 0)
        else:
            states = encoder(**inputs).last_hidden_state
        if self.readout == 'pooled':
            states = (states*mask.unsqueeze(-1)).sum(1)/mask.sum(1,keepdim=True).clamp_min(1)
            mask = mask.any(1)
        return Latent(states,self.output_space,mask=mask,sources=tuple(sources),
                      metadata={'representation':'native_encoder_states','readout':self.readout,
                                'foundation':self.config.get('foundation'),
                                'readout_initialization':'untrained' if self.readout == 'output_encoding' else 'native'})



class _TextObjective(Operation):
    replayable = True

    def __init__(self, owner):
        import weakref
        self._owner = weakref.ref(owner)

    def forward(self, value, *, context=None):
        if not isinstance(value,Mapping) or set(value) != {'inputs','targets'}:
            raise ValueError('objective envelope requires inputs and targets')
        inputs=value['inputs']
        if isinstance(inputs,Mapping):
            if set(inputs) != {'value','context'}:
                raise ValueError('conditioning envelope requires exactly value and context')
            if context:
                raise ValueError('context must appear only inside the conditioning envelope')
            context=inputs['context']
            inputs=inputs['value']
        return self._owner().loss(inputs,value['targets'],context=context).clone()

    def parameters(self, recurse=True):
        return self._owner().parameters(recurse=recurse)

    def _operation_identity(self):
        return self._owner()._tool_identity() + '.objective'

    def configuration(self):
        owner = self._owner()
        return {'operation':type(owner).__module__+'.'+type(owner).__qualname__,
                'role':'objective','model':owner.configuration()}


class TextDecoder(LatentOperation):
    """Project latent sequences into native encoder *input embeddings*.

    The foundation encoder actually processes that sequence before decoding.
    A learned bridge starts untrained. Identity bridging requires the exact
    explicitly declared native input-embedding Space, not merely equal width.
    """
    replayable = False
    training_inputs_include_targets = True

    def __init__(self, config):
        super().__init__(config)
        allowed = {'native_config', 'native_parameter_aliases', 'native_generation_config',
                   'tokenizer', 'input_space', 'native_input_space', 'bridge',
                   'bridge_training', 'generation', 'foundation'}
        if set(config) - allowed:
            raise ValueError(f'Unknown configuration fields: {sorted(set(config) - allowed)}')
        from transformers import AutoModelForSeq2SeqLM, GenerationConfig
        native = _native_config(config['native_config'])
        self.model = AutoModelForSeq2SeqLM.from_config(native)
        _restore_parameter_aliases(self.model,config.get("native_parameter_aliases"))
        if 'native_generation_config' in config:
            self.model.generation_config=GenerationConfig.from_dict(config['native_generation_config'])
        self.tokenizer = _tokenizer(config['tokenizer'])
        self.input_space = Space(**config['input_space'])
        self.native_input_space = Space(**config['native_input_space'])
        bridge = config.get('bridge','linear')
        if bridge == 'identity':
            if self.input_space != self.native_input_space:
                raise ValueError('identity bridge requires the explicit native input embedding space')
            self.projection = nn.Identity()
        elif bridge == 'linear':
            self.projection = nn.Linear(self.input_space.dimensions,_width(native))
        else:
            raise ValueError('bridge must be linear or identity')
        self.generation = {'max_new_tokens':32,'do_sample':False,**config.get('generation',{})}
        if set(self.generation)-{'max_new_tokens','min_new_tokens','num_beams','do_sample','temperature','top_k','top_p','repetition_penalty','length_penalty','early_stopping'}:
            raise ValueError('unsupported generation setting')
        self.training_operation = _TextObjective(self)

    @classmethod
    def from_foundation(cls,repo,*,input_space,revision=None,bridge='linear',generation=None,**kwargs):
        model,tokenizer = _load_foundation(repo,revision,kwargs,decoder=True)
        native_space = Space(f'{repo}:encoder:input_embeddings',_width(model.config),version=revision or 'unversioned',organization='sequence')
        config={'native_config':json.loads(model.config.to_json_string()),'native_parameter_aliases':_parameter_aliases(model),'native_generation_config':json.loads(model.generation_config.to_json_string()),'tokenizer':_tokenizer_config(tokenizer),
                'input_space':input_space.configuration(),'native_input_space':native_space.configuration(),
                'bridge':bridge,'bridge_training':'native_identity' if bridge=='identity' else 'untrained',
                'generation':generation or {'max_new_tokens':32},'foundation':{'repo':str(repo),'revision':revision}}
        with torch.device("meta"):
            result=cls(config)
        result.model=model
        param=next(model.parameters())
        if bridge == "linear":
            result.projection=nn.Linear(input_space.dimensions,_width(model.config),device=param.device,dtype=param.dtype)
        return result.eval()

    @property
    def replayable(self):
        return not self.generation.get('do_sample',False)

    def configuration(self):
        config=super().configuration()
        config['tokenizer']=_tokenizer_config(self.tokenizer)
        config['generation']=dict(self.generation)
        config['native_generation_config']=json.loads(self.model.generation_config.to_json_string())
        return self._validated_config(config)

    def operation_bindings(self):
        return {**super().operation_bindings(),'objective':self.training_operation}

    def embed_text(self,value):
        """Expose actual native token embeddings with their exact input Space."""
        texts=[value] if isinstance(value,str) else value
        if not isinstance(texts,(list,tuple)) or not texts or not all(isinstance(t,str) for t in texts):
            raise ValueError('text input must be a string or nonempty list of strings')
        tokens=self.tokenizer(texts,padding=True,return_tensors='pt')
        device=next(self.model.parameters()).device
        embeddings=self.model.get_encoder().get_input_embeddings()(tokens['input_ids'].to(device))
        return Latent(embeddings,self.native_input_space,mask=tokens['attention_mask'].to(device).bool(),
                      metadata={'representation':'native_input_embeddings'})

    def _inputs(self,value,context):
        values=[*_context(context,'latents'),value]
        pairs=[as_sequence(v,self.input_space) for v in values]
        if len({t.shape[0] for t,m in pairs}) != 1:
            raise ValueError('context latents must have the same batch size')
        tensor=torch.cat([t for t,m in pairs],dim=1)
        mask=torch.cat([m for t,m in pairs],dim=1)
        if not mask.any(1).all():
            raise ValueError('every input sequence must contain an unmasked position')
        # Masking attention alone leaves positional gaps between valid inputs.
        # Pack before projection so masked values cannot affect its gradients.
        lengths=mask.sum(1)
        tensor=nn.utils.rnn.pad_sequence(
            [row[keep] for row,keep in zip(tensor,mask)],batch_first=True)
        mask=torch.arange(tensor.shape[1],device=mask.device)[None,:] < lengths[:,None]
        param=next(self.model.parameters())
        tensor=tensor.to(device=param.device,dtype=param.dtype)
        return self.projection(tensor),mask.to(param.device)

    def forward(self,value,*,context=None):
        embeds,mask=self._inputs(value,context)
        modes={module:module.training for module in self.model.modules()}
        try:
            self.model.eval()
            ids=self.model.generate(inputs_embeds=embeds,attention_mask=mask,**self.generation)
        finally:
            for module,mode in modes.items():
                module.training=mode
        texts=self.tokenizer.batch_decode(ids,skip_special_tokens=True)
        return texts[0] if len(texts)==1 else texts

    def loss(self,value,targets,*,context=None):
        """Differentiable teacher forcing; target text is never encoder context."""
        embeds,mask=self._inputs(value,context)
        texts=[targets] if isinstance(targets,str) else targets
        if not isinstance(texts,(list,tuple)) or len(texts)!=embeds.shape[0] or not all(isinstance(t,str) for t in texts):
            raise ValueError('targets must contain one string per input batch row')
        tokens=self.tokenizer(texts,padding=True,return_tensors='pt')
        labels=tokens['input_ids'].to(embeds.device)
        labels=labels.masked_fill(~tokens['attention_mask'].to(embeds.device).bool(),-100)
        return self.model(inputs_embeds=embeds,attention_mask=mask,labels=labels).loss
