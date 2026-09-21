import importlib.util
from pathlib import Path
import runpy

import pytest
import torch


def runner():
    path=Path(__file__).parents[2]/'.development/experiments/probe_generative_quality.py'
    spec=importlib.util.spec_from_file_location('generative_quality',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_quality_prompts_exclude_targets_and_preserve_exact_inputs():
    mod=runner()
    row={'question':'Which city?', 'candidate':'Rome', 'evidence':[{'source_id':'a','text':'The city is Rome.'}], 'targets':{'support':False},'reference_answer':'SECRET'}
    prompts=mod.prompts(row)
    assert set(prompts)=={'support','completeness','constraints'}
    assert all('SECRET' not in p and 'targets' not in p for p in prompts.values())
    assert all('Rome' in p and 'Which city?' in p and 'The city is Rome.' in p for p in prompts.values())


def test_conditional_yes_no_scores_equal_native_decoder_logits():
    from tensorcode.tools.chatbot import Chatbot
    config=runpy.run_path(str(Path(__file__).parents[1]/'models/test_chatbot_model.py'))['tiny_config']()
    config['max_input_tokens']=256
    model=Chatbot(config).eval()
    mod=runner()
    row={'question':'hello', 'candidate':'world', 'evidence':[{'source_id':'a','text':'hello world'}]}
    # Tiny real transformer; explicit label IDs isolate the scoring mechanism.
    result=mod.assess(model,row,yes_id=5,no_id=6)
    prompt=mod.prompts(row)['support']
    encoded=model.tokenizer(prompt,return_tensors='pt')
    with torch.no_grad():
        logits=model.foundation(input_ids=encoded['input_ids'],attention_mask=encoded['attention_mask'],decoder_input_ids=torch.tensor([[0]])).logits[0,0,[6,5]]
    expected=torch.softmax(logits.float(),dim=-1)[1].item()
    assert abs(result['scores']['support']-expected)<1e-6
    assert not result['input_truncated']
    model.config['max_input_tokens']=2
    refused=mod.assess(model,row,yes_id=5,no_id=6)
    assert refused['input_truncated'] and refused['scores'] is None


def test_workspace_training_owns_only_adapter_updates_and_replays_exactly(tmp_path):
    from tensorcode.tools.chatbot import Chatbot
    from tensorcode.training import ToolTrainer
    config=runpy.run_path(str(Path(__file__).parents[1]/'models/test_chatbot_model.py'))['tiny_config']()
    model=Chatbot(config)
    trainer=runner().workspace_trainer(model,lr=.001)
    before={k:v.clone() for k,v in model.foundation.state_dict().items()}
    experience=trainer.capture(['hello world'],['answer'],source='authored mechanism fixture')
    trainer.step(experience)
    assert model.memory_gate.grad is not None
    assert all(p.grad is None for p in model.foundation.parameters())
    assert all(torch.equal(v,model.foundation.state_dict()[k]) for k,v in before.items())
    trainer.save_checkpoint(tmp_path/'checkpoint',progress={'epochs':1})
    expected_loss=trainer.step(experience)
    expected={k:v.clone() for k,v in model.state_dict().items()}
    restored=Chatbot(config)
    resumed=runner().workspace_trainer(restored,lr=.001)
    resumed.load_checkpoint(tmp_path/'checkpoint')
    replay=resumed.capture(['hello world'],['answer'],source='authored mechanism fixture')
    assert resumed.step(replay)==expected_loss
    assert all(torch.equal(v,restored.state_dict()[k]) for k,v in expected.items())


def test_training_runner_roundtrip_on_tiny_owned_model(tmp_path):
    from tensorcode.tools.chatbot import Chatbot
    import json
    config=runpy.run_path(str(Path(__file__).parents[1]/'models/test_chatbot_model.py'))['tiny_config']()
    config['max_input_tokens']=256
    vocabulary=json.loads(config['tokenizer_json'])
    vocabulary['model']['vocab'].update(yes=8,no=9)
    config['tokenizer_json']=json.dumps(vocabulary);config['foundation_config']['vocab_size']=10
    model=Chatbot(config).eval().requires_grad_(False)
    path=Path(__file__).parents[2]/'examples/train_response_quality.py'
    spec=importlib.util.spec_from_file_location('quality_train_helper',path)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    row={'id':'fixture','question':'hello','candidate':'world','evidence':[{'id':'a','text':'hello world'}],
         'targets':{'support':True,'completeness':False,'constraints':None}}
    result=runner().train_workspace(model,[row],helper,tmp_path,epochs=1,batch_size=2,lr=.001)
    assert result['foundation_unchanged'] and result['optimizer_continuation_exact']
    assert result['steps']==1 and result['supervised_axis_examples']==2
    assert result['adapter_parameters_changed']>0
    restored=Chatbot.from_pretrained(tmp_path/'model')
    assert all(torch.equal(v,restored.state_dict()[k]) for k,v in model.state_dict().items())


def test_streaming_state_digest_covers_nested_optimizer_values():
    mod=runner()
    state={'state':{0:{'step':torch.tensor(1.),'exp_avg':torch.tensor([1.,2.],dtype=torch.bfloat16)}},'groups':[{'lr':.001}]}
    first=mod.state_digest(state)
    import copy
    assert mod.state_digest(copy.deepcopy(state))==first
    state['state'][0]['exp_avg'][1]=3
    assert mod.state_digest(state)!=first
    assert mod.state_digest({'x':[1,2]})!=mod.state_digest({'x':(1,2)})


@pytest.mark.parametrize('autocast_dtype',[None,torch.bfloat16])
def test_foundation_training_updates_native_weights_and_continues_exactly(tmp_path,autocast_dtype,monkeypatch):
    from tensorcode.tools.chatbot import Chatbot
    import json
    config=runpy.run_path(str(Path(__file__).parents[1]/'models/test_chatbot_model.py'))['tiny_config']()
    config['max_input_tokens']=256
    vocabulary=json.loads(config['tokenizer_json'])
    vocabulary['model']['vocab'].update(yes=8,no=9)
    config['tokenizer_json']=json.dumps(vocabulary);config['foundation_config']['vocab_size']=10
    model=Chatbot(config).eval().requires_grad_(False)
    path=Path(__file__).parents[2]/'examples/train_response_quality.py'
    spec=importlib.util.spec_from_file_location('quality_train_helper',path)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    row={'id':'fixture','question':'hello','candidate':'world','evidence':[{'id':'a','text':'hello world'}],
         'targets':{'support':True,'completeness':False,'constraints':None}}
    mod=runner()
    before=mod.state_digest(model.foundation.state_dict())
    native_ids={id(p) for p in model.foundation.parameters()}
    original_init=torch.optim.AdamW.__init__
    groups=[]
    def checked_init(optimizer,parameters,**options):
        original_init(optimizer,parameters,**options)
        assert optimizer.param_groups[0]['lr']==2e-5
        assert optimizer.param_groups[1]['lr']==.001
        assert {id(p) for p in optimizer.param_groups[0]['params']}==native_ids
        assert not native_ids & {id(p) for p in optimizer.param_groups[1]['params']}
        groups.append(len(optimizer.state))
    monkeypatch.setattr(torch.optim.AdamW,'__init__',checked_init)
    result=mod.train_workspace(model,[row],helper,tmp_path,epochs=1,batch_size=2,lr=.001,
                               train_foundation=True,foundation_lr=2e-5,autocast_dtype=autocast_dtype)
    assert not result['foundation_unchanged'] and result['optimizer_continuation_exact']
    assert result['foundation_sha256_before']!=result['foundation_sha256_after']
    assert before!=mod.state_digest(model.foundation.state_dict())
    assert result['foundation_lr']==2e-5 and result['steps']==1
    assert groups==[0,0,0]  # Both restores reconstruct an empty optimizer first.
    assert all(p.dtype==torch.float32 for p in model.parameters())
    restored=Chatbot.from_pretrained(tmp_path/'model')
    assert mod.state_digest(model.state_dict())==mod.state_digest(restored.state_dict())
    expected=mod.assess(model,helper.model_inputs(row),yes_id=8,no_id=9,workspace_ablation=None,autocast_dtype=autocast_dtype)
    actual=mod.assess(restored,helper.model_inputs(row),yes_id=8,no_id=9,workspace_ablation=None,autocast_dtype=autocast_dtype)
    assert expected==actual


@pytest.mark.parametrize('autocast_dtype',[None,torch.bfloat16])
def test_capture_without_gradients_replays_same_foundation_update(autocast_dtype):
    from contextlib import nullcontext
    from tensorcode.tools.chatbot import Chatbot
    from tensorcode.training import ToolTrainer
    config=runpy.run_path(str(Path(__file__).parents[1]/'models/test_chatbot_model.py'))['tiny_config']()
    first=Chatbot(config)
    second=Chatbot(config);second.load_state_dict(first.state_dict())
    losses=[]
    mod=runner()
    for model,capture_context in ((first,nullcontext()),(second,torch.no_grad())):
        trainer=ToolTrainer(model,optimizer=lambda ps:torch.optim.AdamW(ps,lr=2e-5,foreach=False))
        model.eval()
        with mod.computation_context(model,autocast_dtype):
            with capture_context:
                experience=trainer.capture(['hello world'],['answer'],source='authored mechanism fixture')
            losses.append(trainer.step(experience))
        assert any(p.grad is not None for p in model.foundation.parameters())
    assert losses[0]==losses[1]
    assert mod.state_digest(first.state_dict())==mod.state_digest(second.state_dict())
