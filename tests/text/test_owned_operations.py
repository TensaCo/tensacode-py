import pytest
from tensorcode.ops import text

@pytest.mark.parametrize('cls',[text.Transform,text.Classify,text.Score,text.Decide,text.Retrieve])
def test_constructor_rejects_provider(cls):
    with pytest.raises((TypeError,ValueError),match='config|JSON'):
        cls(lambda messages: 'answer')


def test_explicit_provider_factory():
    op=text.Transform.from_model(lambda messages:'answer')
    assert op((text.Message('user','question'),))[-1].content=='answer'
    assert not op.replayable
    with pytest.raises(ValueError,match='external'):
        op.save_pretrained('unused')

@pytest.fixture
def foundation(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, T5Config, T5ForConditionalGeneration
    backend=Tokenizer(models.WordLevel({'[PAD]':0,'[UNK]':1,'question':2,'answer':3},unk_token='[UNK]'))
    backend.pre_tokenizer=pre_tokenizers.Whitespace()
    tokenizer=PreTrainedTokenizerFast(tokenizer_object=backend,pad_token='[PAD]',unk_token='[UNK]')
    model=T5ForConditionalGeneration(T5Config(vocab_size=4,d_model=8,d_ff=16,num_layers=1,num_decoder_layers=1,num_heads=2,dropout_rate=0.,decoder_start_token_id=0,eos_token_id=3,pad_token_id=0))
    path=tmp_path/'native'
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    return path

@pytest.mark.parametrize('cls,options,target',[
    (text.Transform,{},'answer'),
    (text.Classify,{'labels':['yes','no']},{'label':'yes','distribution':None,'confidence':None,'abstained':False}),
    (text.Score,{'rubric':['bad','good']},{'score':1,'distribution':None,'confidence':None,'abstained':False}),
    (text.Decide,{'options':['yes','no']},{'choice':'yes','distribution':None,'confidence':None,'abstained':False}),
    (text.Retrieve,{'items':{'a':'answer'}},{'keys':['a'],'scores':None,'abstained':False}),
])
def test_owned_loss_artifact_and_training_restart(foundation,tmp_path,cls,options,target):
    import json
    import torch
    from tensorcode.training import Trainer
    from tensorcode.training import load_experience
    op=cls.from_foundation(foundation,config={**options,'generation':{'max_new_tokens':2}})
    value=(text.Message('user','question'),)
    seen=[]
    hook=op.model.model.get_encoder().register_forward_pre_hook(lambda module,args,kwargs:seen.append(kwargs['input_ids'].clone()),with_kwargs=True)
    loss=op.loss(value,target)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in op.parameters())
    hook.remove()
    torch.testing.assert_close(seen[0],op.model.inputs(op._request(value,None))['input_ids'])
    path=tmp_path/'owned'
    op.save_pretrained(path)
    manifest=json.loads((path/'tensorcode_config.json').read_text())
    assert manifest['tool']==cls.__module__+'.'+cls.__qualname__
    restored=cls.from_pretrained(path)
    torch.testing.assert_close(restored.loss(value,target),loss)
    assert restored.configuration()==op.configuration()
    trainer=Trainer.from_tool(restored,lr=.001)
    session=trainer.capture(value,target,source='authored-test')
    session.save(tmp_path/'session.json',operations=trainer.operations,codecs={'message':text.Message})
    session=load_experience(tmp_path/'session.json',operations=trainer.operations,codecs={'message':text.Message})
    trainer.step(session)
    trainer.save_checkpoint(tmp_path/'checkpoint',progress={'batch':1})
    trainer.step(session)
    expected={k:v.clone() for k,v in restored.state_dict().items()}
    assert trainer.load_checkpoint(tmp_path/'checkpoint')=={'batch':1}
    trainer.step(session)
    for key,value in expected.items():
        torch.testing.assert_close(restored.state_dict()[key],value,rtol=0,atol=0)


def test_owned_structured_targets_and_generation_are_strict(foundation):
    op=text.Classify.from_foundation(foundation,config={'labels':['yes','no'],'generation':{'max_new_tokens':1}})
    value=(text.Message('user','question'),)
    with pytest.raises(ValueError,match='mapping'):
        op.loss(value,'yes')
    with pytest.raises(text.InvalidModelOutput,match='fields'):
        op.loss(value,{'label':'yes','abstained':False})
    with pytest.raises(text.InvalidModelOutput,match='configured labels'):
        op.loss(value,{'label':'other','distribution':None,'confidence':None,'abstained':False})
    with pytest.raises(text.InvalidModelOutput):
        op(value)  # Tiny random model cannot manufacture a valid configured JSON answer.


def test_owned_config_rejects_unknown_fields_and_class_mismatch(foundation,tmp_path):
    op=text.Transform.from_foundation(foundation)
    with pytest.raises(ValueError,match='Unknown'):
        text.Transform({**op.configuration(),'old_model':'discarded'})
    op.save_pretrained(tmp_path/'owned')
    with pytest.raises(ValueError,match='tool'):
        text.Classify.from_pretrained(tmp_path/'owned')

@pytest.mark.parametrize('cls,config',[(text.TextEncoder,{}),(text.TextDecoder,{}),(text.ImageEncoder,{'detail':'high'})])
def test_pure_config_roundtrip(cls,config,tmp_path):
    op=cls(config)
    op.save_pretrained(tmp_path/'pure')
    assert cls.from_pretrained(tmp_path/'pure').configuration()==op.configuration()
    with pytest.raises(ValueError):
        cls({'model':'obsolete'})


def test_external_transform_configuration_tracks_instructions():
    provider=lambda messages:'answer'
    first=text.Transform.from_model(provider,instructions='Summarize')
    second=text.Transform.from_model(provider,instructions='Translate')
    assert first.configuration()!=second.configuration()


@pytest.mark.parametrize('second_call',['complete','loss'])
def test_native_generation_serializes_mode_sensitive_calls(foundation,monkeypatch,second_call):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    import torch
    from types import SimpleNamespace
    op=text.Transform.from_foundation(foundation).train()
    request=op._request((text.Message('user','question'),),None)
    first_entered=threading.Event()
    second_entered=threading.Event()
    second_started=threading.Event()
    release_first=threading.Event()
    count=0
    def generate(**kwargs):
        nonlocal count
        count+=1
        if count==1:
            first_entered.set()
            assert release_first.wait(2)
        else:
            second_entered.set()
        return torch.tensor([[3]])
    def forward(**kwargs):
        second_entered.set()
        return SimpleNamespace(loss=torch.tensor(1.))
    monkeypatch.setattr(op.model.model,'generate',generate)
    monkeypatch.setattr(op.model.model,'forward',forward)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first=pool.submit(op.model.complete,request)
        assert first_entered.wait(2)
        def run_second():
            second_started.set()
            return op.model.complete(request) if second_call=='complete' else op.model.loss(request,'answer')
        second=pool.submit(run_second)
        assert second_started.wait(2)
        try:
            assert not second_entered.wait(.05), 'shared native state was used during temporary evaluation mode'
        finally:
            release_first.set()
        first.result()
        second.result()
    assert op.model.model.training
    assert second_entered.is_set()
