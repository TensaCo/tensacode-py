import importlib.util
from pathlib import Path
import runpy

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
