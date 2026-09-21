"""Tiny local encoders test ownership and pooling, not pretrained retrieval quality."""
import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

from tensorcode._internal.retrieval import RetrievalEncoder
from tensorcode.runtime.cognitive_state import Evidence
from tensorcode.runtime.cognition import LearnedEpisodicMemory
from tensorcode.tools.investigator import Investigator


def retrieval_config():
    tokenizer = Tokenizer(models.WordLevel({'<pad>':0,'<unk>':1,'alpha':2,'beta':3},unk_token='<unk>'))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    return {'foundation_config':BertConfig(vocab_size=4,hidden_size=8,num_hidden_layers=1,num_attention_heads=2,intermediate_size=16,hidden_dropout_prob=0,attention_probs_dropout_prob=0).to_dict(),
            'tokenizer_json':tokenizer.to_str(),
            'tokenizer_special_tokens':{'pad_token':'<pad>','unk_token':'<unk>'},
            'pooling':'masked_mean','normalize':True,'max_tokens':4}


def test_owned_masked_mean_has_no_projection_and_padding_does_not_change_embedding():
    model = RetrievalEncoder(retrieval_config()).eval()
    parameters = {id(p) for p in model.parameters()}
    alone = model(['alpha'])
    together = model(['alpha','beta beta beta'])
    assert alone.shape == (1,8) and together.dtype == torch.float32
    assert torch.allclose(alone[0],together[0],atol=1e-6)
    assert torch.allclose(together.norm(dim=-1),torch.ones(2),atol=1e-6)
    assert parameters == {id(p) for p in model.parameters()}
    assert {id(p) for p in model.parameters()} == {id(p) for p in model.encode.parameters()}
    receipt = model.receipt(['alpha alpha alpha alpha alpha'])
    assert receipt['input_truncated'] == [True]
    assert receipt['pooling'] == 'masked_mean' and receipt['normalized']
    assert receipt['dimensions'] == 8
    for invalid in ({'pooling':'cls'},{'normalize':False},{'generator':{}}):
        with pytest.raises(ValueError): RetrievalEncoder(dict(retrieval_config(),**invalid))


def test_explicit_contrastive_targets_produce_gradients_without_new_parameters():
    model = RetrievalEncoder(retrieval_config())
    parameters = {id(p) for p in model.parameters()}
    loss = model.loss(['alpha','beta'],['alpha','beta'],torch.eye(2,dtype=torch.bool))
    assert loss.isfinite()
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encode.parameters())
    assert parameters == {id(p) for p in model.parameters()}
    with pytest.raises(ValueError): model.loss(['alpha'],['beta'],[[False]])


def test_investigator_owns_retrieval_artifact_and_memory_fingerprints_only_that_encoder(tmp_path):
    tool = Investigator({'vocabulary':['alpha','beta'],'dimensions':4,'slots':2,'steps':1,
                         'retrieval_encoder':retrieval_config()}).eval()
    assert tool.episodic_encoder is not None
    assert 'episodic_encoder.encode' in tool.operation_bindings()
    memory = LearnedEpisodicMemory(tool)
    memory.remember(Evidence('a','alpha','document'),episode_id='past')
    assert memory.metadata['encoder'] == 'owned_retrieval_encoder'
    fingerprint = memory.fingerprint
    with torch.no_grad(): tool.rank.encode.module.weight.add_(1)
    assert memory.fingerprint == fingerprint
    assert memory.retrieve('alpha')[0].score == pytest.approx(1)
    expected = tool.episodic_encoder(['alpha','beta']).detach()
    tool.save_pretrained(tmp_path/'model')
    restored = Investigator.from_pretrained(tmp_path/'model')
    assert torch.equal(restored.episodic_encoder(['alpha','beta']),expected)
    rebuilt = LearnedEpisodicMemory.from_snapshot(memory.snapshot(),investigator=restored)
    assert rebuilt.retrieve('alpha')[0].evidence.source_id == 'document'
    with torch.no_grad(): next(tool.episodic_encoder.parameters()).add_(.1)
    with pytest.raises(ValueError,match='stale'): memory.retrieve('alpha')
    memory.rebuild_index()
    assert memory.retrieve('alpha')[0].score == pytest.approx(1)


def test_explicit_local_foundation_bootstrap_copies_encoder_and_tokenizer(tmp_path):
    config = retrieval_config(); root = tmp_path/'foundation'
    native = BertModel(BertConfig(**config['foundation_config'])).eval()
    native.save_pretrained(root)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer.from_str(config['tokenizer_json']),**config['tokenizer_special_tokens'])
    tokenizer.save_pretrained(root)
    owned = RetrievalEncoder.from_foundation(root,pooling='masked_mean',normalize=True,max_tokens=4,local_files_only=True)
    assert torch.equal(next(owned.encode.module.model.parameters()),next(native.parameters()))
    assert owned.configuration()['foundation']['repository'] == str(root)
    tool = Investigator.from_retrieval_foundation(root,pooling='masked_mean',normalize=True,max_tokens=4,
                    local_files_only=True,vocabulary=['alpha','beta'],dimensions=4,slots=2,steps=1)
    assert torch.equal(tool.episodic_encoder(['alpha']),owned(['alpha']))


def test_retrieval_tooltrainer_trace_roundtrip_and_optimizer_invalidates_index(tmp_path):
    from tensorcode import training
    tool = Investigator({'vocabulary':['alpha','beta'],'dimensions':4,'slots':2,'steps':1,
                         'retrieval_encoder':retrieval_config()}).eval()
    memory = LearnedEpisodicMemory(tool)
    memory.remember(Evidence('a','alpha','source'),episode_id='past')
    inputs = {'queries':['alpha','beta'],'documents':['alpha','beta']}
    targets = [[True,False],[False,True]]
    trainer = training.ToolTrainer(tool,lr=.01)
    experience = trainer.capture({'mode':'retrieval','inputs':inputs},targets,source='authored-mechanism-fixture')
    before = next(tool.episodic_encoder.parameters()).detach().clone()
    assert trainer.step(experience) >= 0
    assert not torch.equal(before,next(tool.episodic_encoder.parameters()))
    with pytest.raises(ValueError,match='stale'): memory.retrieve('alpha')
    experience.save(tmp_path/'experience.json',operations=trainer.operations)
    tool.save_pretrained(tmp_path/'model')
    trainer.save_checkpoint(tmp_path/'resume')
    restored = training.ToolTrainer(Investigator.from_pretrained(tmp_path/'model'),lr=.01)
    restored.load_checkpoint(tmp_path/'resume')
    loaded = training.load(tmp_path/'experience.json',operations=restored.operations)
    assert restored.step(loaded) >= 0
    with pytest.raises(ValueError): tool.retrieval_loss(dict(inputs,targets=targets),targets)
    assert tool.episodic_encoder.contrastive_loss(inputs['queries'],inputs['documents'],targets).isfinite()


def test_rank_trace_roundtrip_normalizes_retrieval_defaults_before_rank_construction(tmp_path):
    from tensorcode import training
    tool = Investigator({'vocabulary':['alpha','beta'],'dimensions':4,'slots':2,'steps':1,
                         'retrieval_encoder':retrieval_config()}).eval()
    trainer = training.ToolTrainer(tool)
    inputs = {'question':'alpha','evidence':[],
              'hypotheses':[{'id':'a','text':'alpha'},{'id':'b','text':'beta'}]}
    experience = trainer.capture(inputs,'a',source='authored-mechanism-fixture')
    experience.save(tmp_path/'rank-experience.json',operations=trainer.operations)
    tool.save_pretrained(tmp_path/'model')
    restored = training.ToolTrainer(Investigator.from_pretrained(tmp_path/'model'))
    loaded = training.load(tmp_path/'rank-experience.json',operations=restored.operations)
    assert restored.step(loaded) >= 0
