"""Owned pairwise vector scoring with an explicit authored score meaning."""
from __future__ import annotations
import torch
from torch import nn
from tensorcode._internal.latent_ops import LatentOperation
from tensorcode._internal.vec.owned import OwnedMap, Objective, positive
from .candidates import CandidateSet, Scores
from .latent import Latent, Space, require_compatible


class _PairReadout(OwnedMap):
    kind = 'decode'


class Score(LatentOperation):
    training_inputs_include_targets = True
    def __init__(self, config):
        super().__init__(config)
        config=self.config
        allowed={'architecture','query_space','candidate_space','meaning','hidden_dimensions','native_config','foundation','pair_dimensions'}
        if set(config)-allowed:
            raise ValueError(f'unknown configuration fields: {sorted(set(config)-allowed)}')
        self.query_space=Space(**config['query_space'])
        self.candidate_space=Space(**config['candidate_space'])
        self.meaning=config['meaning']
        if not isinstance(self.meaning,str) or not self.meaning.strip():
            raise ValueError('Score meaning must be a nonempty string')
        width=positive(config.get('pair_dimensions',min(self.query_space.dimensions,self.candidate_space.dimensions)), 'pair_dimensions')
        self.config['pair_dimensions']=width
        self.query_projection=nn.Linear(self.query_space.dimensions,width)
        self.candidate_projection=nn.Linear(self.candidate_space.dimensions,width)
        self.pair_space=Space('tensorcode:score:interacting-pair',width*3)
        readout={k:v for k,v in config.items() if k in {'architecture','hidden_dimensions','native_config','foundation'}}
        readout.update(input_space=self.pair_space.configuration(),output_dimensions=1,output=self.meaning)
        self.module=_PairReadout(readout)
        self.config.update(query_space=self.query_space.configuration(),candidate_space=self.candidate_space.configuration())
        self.config['architecture']=self.module.config['architecture']
        if 'native_config' in readout: self.config['native_config']=self.module.config['native_config']
        self._supplied=False
        self.objective=Objective(self)

    @classmethod
    def from_module(cls,module,*,query_space,candidate_space,meaning):
        if not isinstance(module,nn.Module): raise TypeError('Score module must be a torch.nn.Module')
        if not isinstance(query_space,Space) or not isinstance(candidate_space,Space): raise TypeError('Score spaces must be Space objects')
        if not isinstance(meaning,str) or not meaning.strip(): raise ValueError('Score meaning must be nonempty')
        self=cls.__new__(cls);nn.Module.__init__(self)
        self.module=module;self.query_space=query_space;self.candidate_space=candidate_space;self.meaning=meaning
        self._supplied=True;self.objective=Objective(self)
        return self

    @classmethod
    def from_foundation(cls,repo,*,query_space,candidate_space,meaning,revision=None,**kwargs):
        q=query_space if isinstance(query_space,Space) else Space(**query_space)
        c=candidate_space if isinstance(candidate_space,Space) else Space(**candidate_space)
        width=positive(kwargs.pop('pair_dimensions',min(q.dimensions,c.dimensions)), 'pair_dimensions')
        pair=Space('tensorcode:score:interacting-pair',width*3)
        module=_PairReadout.from_foundation(repo,input_space=pair,output_dimensions=1,output=meaning,revision=revision,**kwargs)
        config={k:v for k,v in module.configuration().items() if k in {'architecture','native_config','foundation'}}
        config.update(query_space=q.configuration(),candidate_space=c.configuration(),meaning=meaning,pair_dimensions=width)
        with torch.device('meta'):
            self=cls(config)
        self.module=module
        parameter=next(module.parameters())
        self.query_projection=nn.Linear(q.dimensions,width,device=parameter.device,dtype=parameter.dtype)
        self.candidate_projection=nn.Linear(c.dimensions,width,device=parameter.device,dtype=parameter.dtype)
        return self.eval()

    def forward(self,value,*,context=None):
        if not isinstance(value,CandidateSet): raise TypeError('Score expects a CandidateSet')
        require_compatible(self.query_space,value.query.space,role='query')
        require_compatible(self.candidate_space,value.candidates.space,role='candidates')
        if self._supplied:
            if context: raise ValueError('supplied Score does not consume context')
            scores=self.module(value.query.tensor,value.candidates.tensor)
        else:
            if context: raise ValueError('Score does not consume context; supply explicit query and candidates')
            query=value.query.tensor
            candidates=value.candidates.tensor
            if query.device != candidates.device or query.dtype != candidates.dtype:
                raise ValueError('query and candidates require matching dtype and device')
            if value.query.mask is not None and (value.query.mask.dtype != torch.bool or not value.query.mask.all()):
                raise ValueError('every query must be valid with a boolean mask')
            query_features=self.query_projection(query).unsqueeze(-2)
            candidate_features=self.candidate_projection(candidates)
            query_features=query_features.expand_as(candidate_features)
            pairs=torch.cat([query_features,candidate_features,query_features*candidate_features],dim=-1)
            scores=self.module(Latent(pairs.reshape(-1,pairs.shape[-1]),self.pair_space)).reshape(candidates.shape[:-1])
            if value.candidates.mask is not None: scores=scores.masked_fill(~value.candidates.mask,0)
        return Scores(scores,self.meaning,value)

    def loss(self,value,targets,*,context=None):
        scores=self.forward(value,context=context).values
        if not isinstance(targets,torch.Tensor) or targets.shape != scores.shape:
            raise ValueError('score targets must match candidate scores')
        targets=targets.to(scores)
        if value.candidates.mask is not None:
            scores=scores[value.candidates.mask];targets=targets[value.candidates.mask]
        if not torch.isfinite(targets).all(): raise ValueError('valid targets must be finite')
        return (scores-targets).square().mean()

    def configuration(self):
        if self._supplied:
            from ._configuration import module_configuration,qualified_name
            return {'operation':qualified_name(self),'query_space':self.query_space.configuration(),'candidate_space':self.candidate_space.configuration(),'meaning':self.meaning,'module':module_configuration(self.module)}
        return super().configuration()

    @property
    def training_operation(self):
        return self.objective

    def operation_bindings(self):
        return {**super().operation_bindings(), 'objective': self.training_operation}

    def save_pretrained(self,directory):
        if self._supplied: raise ValueError('supplied modules have no declarative reconstruction; cannot save_pretrained')
        return super().save_pretrained(directory)
