import pytest
import torch
from tensorcode.ops.vec import Transform, Classify, Score, Decode, Latent, Space, CandidateSet

S = Space('owned-input', 3)
O = Space('owned-output', 2)

def test_owned_linear_transform_roundtrip_and_loss(tmp_path):
    op = Transform({'architecture':'linear','input_space':S.configuration(),'output_space':O.configuration()})
    value = Latent(torch.randn(2,3),S)
    target = Latent(torch.randn(2,2),O)
    loss = op.loss(value,target)
    loss.backward()
    assert all(p.grad is not None for p in op.parameters())
    op.save_pretrained(tmp_path/'model')
    loaded = Transform.from_pretrained(tmp_path/'model')
    assert loaded.configuration() == op.configuration()
    torch.testing.assert_close(loaded(value).tensor,op(value).tensor,rtol=0,atol=0)
    torch.testing.assert_close(loaded.loss(value,target),loss,rtol=0,atol=0)

@pytest.mark.parametrize('cls',[Transform,Classify,Score,Decode])
def test_old_module_constructors_are_rejected(cls):
    with pytest.raises((ValueError,TypeError)):
        cls(torch.nn.Linear(3,2))

def test_owned_classification_and_decode(tmp_path):
    value=Latent(torch.randn(2,3),S)
    for cls,extra in [(Classify,{'labels':['a','b']}),(Decode,{'output_dimensions':2,'output':'regression values'})]:
        op=cls({'architecture':'mlp','input_space':S.configuration(),'hidden_dimensions':[4],**extra})
        targets=torch.tensor([0,1]) if cls is Classify else torch.randn(2,2)
        loss=op.loss(value,targets)
        loss.backward()
        assert all(p.grad is not None for p in op.parameters())
        op.save_pretrained(tmp_path/cls.__name__)
        loaded=cls.from_pretrained(tmp_path/cls.__name__)
        torch.testing.assert_close(loaded.loss(value,targets),loss,rtol=0,atol=0)

def test_owned_pair_score_artifact(tmp_path):
    value=CandidateSet(Latent(torch.randn(3),S),Latent(torch.randn(4,3),S),('a','b','c','d'))
    op=Score({'architecture':'mlp','query_space':S.configuration(),'candidate_space':S.configuration(),'hidden_dimensions':[5],'meaning':'authored relevance logits'})
    loss=op.loss(value,torch.randn(4));loss.backward()
    assert all(p.grad is not None for p in op.parameters())
    op.save_pretrained(tmp_path/'score')
    loaded=Score.from_pretrained(tmp_path/'score')
    torch.testing.assert_close(loaded(value).values,op(value).values,rtol=0,atol=0)

def test_expert_module_artifact_rejected(tmp_path):
    op=Transform.from_module(torch.nn.Linear(3,2))
    assert op(torch.ones(3)).shape == (2,)
    with pytest.raises(ValueError,match='supplied|reconstruct'):
        op.save_pretrained(tmp_path/'unsupported')

def test_native_transformer_context_mask_and_artifact(tmp_path):
    seq=Space('tokens',3,organization='sequence')
    out=Space('states',2,organization='sequence')
    op=Transform({'architecture':'transformer','input_space':seq.configuration(),'output_space':out.configuration(),'native_config':{'model_type':'bert','hidden_size':4,'num_hidden_layers':1,'num_attention_heads':2,'intermediate_size':6,'hidden_dropout_prob':0.0,'attention_probs_dropout_prob':0.0,'vocab_size':8}}).eval()
    x=Latent(torch.randn(2,3),seq)
    prefix=Latent(torch.randn(2,3),seq,mask=torch.tensor([True,False]))
    result=op(x,context={'latents':[prefix]})
    changed=Latent(prefix.tensor+torch.tensor([[0.],[999.]]),seq,mask=prefix.mask)
    torch.testing.assert_close(result.tensor,op(x,context={'latents':[changed]}).tensor)
    assert result.tensor.shape == (2,2)
    op.loss(x,Latent(torch.randn(2,2),out),context={'latents':[prefix]}).backward()
    op.save_pretrained(tmp_path/'native')
    loaded=Transform.from_pretrained(tmp_path/'native')
    torch.testing.assert_close(result.tensor,loaded(x,context={'latents':[prefix]}).tensor,atol=0,rtol=0)
    with pytest.raises(ValueError,match='context'):
        op(x,context={'targets':x})

def test_linear_score_has_query_candidate_interaction():
    space=Space('scalar',1)
    op=Score({'architecture':'linear','query_space':space.configuration(),'candidate_space':space.configuration(),'meaning':'pair utility'})
    first=CandidateSet(Latent(torch.tensor([1.]),space),Latent(torch.tensor([[1.],[2.]]),space),('a','b'))
    second=CandidateSet(Latent(torch.tensor([3.]),space),first.candidates,first.identities)
    a=op(first).values.diff()
    b=op(second).values.diff()
    assert not torch.allclose(a,b)

def test_foundation_native_encoder_weights_survive_owned_artifact(tmp_path):
    from transformers import BertConfig, BertModel
    native=BertModel(BertConfig(hidden_size=4,num_hidden_layers=1,num_attention_heads=2,intermediate_size=6,vocab_size=8))
    native.save_pretrained(tmp_path/'foundation')
    op=Transform.from_foundation(tmp_path/'foundation',input_space=S,output_space=O,local_files_only=True)
    for name,p in native.named_parameters():
        torch.testing.assert_close(p,dict(op.model.named_parameters())[name],atol=0,rtol=0)
    assert op.configuration()['foundation']['input_bridge']=='untrained'
    value=Latent(torch.randn(2,3),S)
    expected=op(value).tensor
    op.save_pretrained(tmp_path/'owned')
    loaded=Transform.from_pretrained(tmp_path/'owned')
    torch.testing.assert_close(expected,loaded(value).tensor,atol=0,rtol=0)
    scorer=Score.from_foundation(tmp_path/'foundation',query_space=S,candidate_space=S,meaning='pair utility',local_files_only=True)
    candidates=CandidateSet(Latent(torch.randn(3),S),Latent(torch.randn(2,3),S),('a','b'))
    scorer.save_pretrained(tmp_path/'scorer')
    loaded_score=Score.from_pretrained(tmp_path/'scorer')
    torch.testing.assert_close(scorer(candidates).values,loaded_score(candidates).values,atol=0,rtol=0)

@pytest.mark.parametrize('architecture',['linear','mlp','transformer'])
def test_unknown_owned_fields_rejected(architecture):
    with pytest.raises(ValueError,match='unknown'):
        Transform({'architecture':architecture,'input_space':S.configuration(),'output_space':O.configuration(),'module':'untrusted'})

def test_masked_regression_targets_do_not_poison_gradients():
    seq=Space('regression-source',3,organization='sequence')
    out=Space('regression-output',2,organization='sequence')
    value=Latent(torch.randn(2,3),seq,mask=torch.tensor([True,False]))
    target=Latent(torch.tensor([[1.,2.],[float('nan'),float('nan')]]),out)
    op=Transform({'input_space':seq.configuration(),'output_space':out.configuration()})
    loss=op.loss(value,target);loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in op.parameters())

def test_sequence_tensor_decode_loss_excludes_source_padding():
    seq=Space('decode-source',3,organization='sequence')
    op=Decode({'input_space':seq.configuration(),'output_dimensions':2,'output':'regression values','readout':'sequence'})
    value=Latent(torch.randn(2,3),seq,mask=torch.tensor([True,False]))
    expected=(op(value)[0]-torch.ones(2)).square().mean()
    targets=torch.tensor([[1.,1.],[100.,100.]])
    torch.testing.assert_close(op.loss(value,targets),expected)

def test_owned_objective_registered_with_public_identity():
    op=Transform({'input_space':S.configuration(),'output_space':O.configuration()})
    assert op.operation_bindings()['objective'] is op.training_operation
    assert op.training_operation._operation_identity()=='tensorcode.ops.vec.transform.Transform.objective'
    value=Latent(torch.randn(2,3),S)
    target=Latent(torch.randn(2,2),O)
    torch.testing.assert_close(op.training_operation({'inputs':value,'targets':target}),op.loss(value,target))

@pytest.mark.parametrize('kind',['transform','classify','decode','score'])
def test_owned_training_exact_restart(kind,tmp_path):
    from tensorcode.training import Trainer, load_experience
    classes={'transform':Transform,'classify':Classify,'decode':Decode,'score':Score}
    configs={
        'transform':{'input_space':S.configuration(),'output_space':O.configuration()},
        'classify':{'input_space':S.configuration(),'labels':['a','b']},
        'decode':{'input_space':S.configuration(),'output_dimensions':2,'output':'regression'},
        'score':{'query_space':S.configuration(),'candidate_space':S.configuration(),'meaning':'utility'},
    }
    value=Latent(torch.randn(2,3),S)
    targets=Latent(torch.randn(2,2),O) if kind=='transform' else torch.tensor([0,1]) if kind=='classify' else torch.randn(2,2)
    if kind=='score':
        value=CandidateSet(value,Latent(torch.randn(2,2,3),S),('a','b'))
    op=classes[kind](configs[kind])
    trainer=Trainer.from_tool(op)
    session=trainer.capture(value,targets,source='authored test targets')
    trainer.step(session)
    trainer.save_checkpoint(tmp_path/'training',progress={'step':1})
    op.save_pretrained(tmp_path/'model')
    codecs={'latent':Latent,'space':Space,'candidates':CandidateSet}
    session.save(tmp_path/'experience.json',operations=trainer.operations,codecs=codecs)
    expected=trainer.step(session)
    state={k:v.clone() for k,v in op.state_dict().items()}
    restored=classes[kind].from_pretrained(tmp_path/'model')
    restarted=Trainer.from_tool(restored)
    replay=load_experience(tmp_path/'experience.json',operations=restarted.operations,codecs=codecs)
    assert restarted.load_checkpoint(tmp_path/'training')=={'step':1}
    assert restarted.step(replay)==expected
    for k,v in restored.state_dict().items():
        torch.testing.assert_close(v,state[k],atol=0,rtol=0)

def test_foundation_factories_construct_replacement_backbones_on_meta(tmp_path,monkeypatch):
    from transformers import AutoModel, BertConfig, BertModel
    native=BertModel(BertConfig(hidden_size=4,num_hidden_layers=1,num_attention_heads=2,intermediate_size=6,vocab_size=8))
    native.save_pretrained(tmp_path/'foundation')
    devices=[]
    original=AutoModel.from_config
    def observe(config,**kwargs):
        model=original(config,**kwargs)
        devices.append(next(model.parameters()).device.type)
        return model
    monkeypatch.setattr(AutoModel,'from_config',observe)
    transform=Transform.from_foundation(tmp_path/'foundation',input_space=S,output_space=O,local_files_only=True)
    score=Score.from_foundation(tmp_path/'foundation',query_space=S,candidate_space=S,meaning='utility',local_files_only=True)
    assert devices and set(devices)=={'meta'}
    assert all(p.device.type=='cpu' for op in (transform,score) for p in op.parameters())
    torch.testing.assert_close(transform.model.embeddings.word_embeddings.weight,native.embeddings.word_embeddings.weight,atol=0,rtol=0)
    torch.testing.assert_close(score.module.model.embeddings.word_embeddings.weight,native.embeddings.word_embeddings.weight,atol=0,rtol=0)
