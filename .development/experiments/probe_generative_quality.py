"""Frozen or workspace-adapted quality judgments on reviewed development data."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

INSTRUCTIONS = {
    'support': 'Based only on the supplied evidence, is the proposed answer to the question supported? Answer yes or no.',
    'completeness': 'Does the proposed answer supply all the information requested by the question, regardless of whether its factual values are correct? Answer yes or no.',
    'constraints': 'Does the proposed answer correctly answer the question while respecting all of its restrictions? Use only the supplied evidence. Answer yes or no.',
}


def prompts(row):
    from tensorcode._internal.response_quality import ResponseQualityAssessor
    value={key:row[key] for key in ('question','candidate','evidence')}
    ResponseQualityAssessor.validate(value)
    # Only the validated inference schema enters the prompt.
    return {axis: instruction+'\n'+json.dumps(value,ensure_ascii=False)
            for axis,instruction in INSTRUCTIONS.items()}


def assess(model,row,*,yes_id,no_id,workspace_ablation='bypass'):
    import torch
    values=prompts(row)
    lengths={axis:len(model.tokenizer(text,truncation=False)['input_ids']) for axis,text in values.items()}
    truncated=any(n>model.config['max_input_tokens'] for n in lengths.values())
    result={'input_token_counts':lengths,'input_truncated':truncated,'scores':None}
    if truncated:
        return result
    with torch.no_grad():
        # Same foundation, explicit native-memory bypass; no learned workspace claim.
        state=model.encode_workspace(list(values.values()),workspace_ablation=workspace_ablation)
        labels=torch.full((len(values),1),yes_id,dtype=torch.long,device=state['conditioning'].device)
        logits=model.decoder(dict(state,labels=labels))['logits'][:,0,:]
        scores=torch.softmax(logits[:,[no_id,yes_id]].float(),dim=-1)[:,1].tolist()
    result['scores']=dict(zip(values,scores))
    return result


def workspace_trainer(model, *, lr):
    import torch
    from tensorcode.training import ToolTrainer
    model.foundation.requires_grad_(False)
    trainer=ToolTrainer(model,optimizer=lambda parameters:torch.optim.AdamW(parameters,lr=lr))
    # Deterministic frozen foundation and adapter computation; eval does not disable gradients.
    model.eval()
    return trainer


def train_workspace(model, rows, helper, output, *, epochs, batch_size, lr):
    import copy
    import random
    import torch
    pairs=[];excluded=[]
    for row in rows:
        values=prompts(helper.model_inputs(row))
        if any(len(model.tokenizer(p,truncation=False)['input_ids'])>model.config['max_input_tokens'] for p in values.values()):
            excluded.append(row['id']);continue
        for axis,prompt in values.items():
            target=row['targets'][axis]
            if target is not None:pairs.append((row['id'],axis,prompt,'yes' if target else 'no'))
    if not pairs:raise ValueError('no eligible supervised prompts')
    # The diagnostic loader froze all parameters; explicitly unfreeze only adapters.
    model.workspace.requires_grad_(True);model.memory_projection.requires_grad_(True)
    model.memory_gate.requires_grad_(True)
    trainer=workspace_trainer(model,lr=lr)
    foundation_before=helper.tensor_digest(model.foundation)
    adapter_before={name:value.detach().clone() for name,value in model.named_parameters() if value.requires_grad}
    rng=random.Random(20260923);losses=[]
    for epoch in range(epochs):
        order=list(pairs);rng.shuffle(order);epoch_losses=[]
        for start in range(0,len(order),batch_size):
            batch=order[start:start+batch_size]
            experience=trainer.capture([r[2] for r in batch],[r[3] for r in batch],source='assistant-reviewed response-quality-v2')
            epoch_losses.append(trainer.step(experience))
            del experience
        losses.append(sum(epoch_losses)/len(epoch_losses))
        print(json.dumps({'epoch':epoch+1,'loss':losses[-1],'steps':trainer.steps}),flush=True)
    trainer.save_checkpoint(output/'training',progress={'epochs':epochs,'supervised_axis_examples':len(pairs)})
    probe=pairs[:batch_size]
    def step():
        experience=trainer.capture([r[2] for r in probe],[r[3] for r in probe],source='continuation probe; restored afterwards')
        return trainer.step(experience)
    expected_loss=step()
    expected={name:value.detach().clone() for name,value in model.named_parameters() if value.requires_grad}
    expected_optimizer=copy.deepcopy(trainer.optimizer.state_dict())
    trainer.load_checkpoint(output/'training')
    actual_loss=step()
    exact=(actual_loss==expected_loss and all(torch.equal(expected[name],value) for name,value in model.named_parameters() if value.requires_grad)
           and helper.state_equal(expected_optimizer,trainer.optimizer.state_dict()))
    if not exact:raise AssertionError('workspace optimizer continuation differed')
    trainer.load_checkpoint(output/'training')
    model.eval();model.save_pretrained(output/'model')
    unchanged=foundation_before==helper.tensor_digest(model.foundation)
    if not unchanged:raise AssertionError('frozen foundation changed')
    return {'epochs':epochs,'batch':batch_size,'lr':lr,'seed':20260923,'loss':'native teacher-forced yes/no sequence cross entropy',
            'supervised_axis_examples':len(pairs),'excluded':excluded,'epoch_losses':losses,'steps':trainer.steps,
            'foundation_unchanged':unchanged,'foundation_sha256':foundation_before,'optimizer_continuation_exact':exact,
            'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'adapter_parameters_changed':sum(not torch.equal(adapter_before[n],v) for n,v in model.named_parameters() if v.requires_grad)}


def run(args):
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    from tensorcode.tools.chatbot import Chatbot
    if not torch.cuda.is_available():raise RuntimeError('real models require authorized CUDA host')
    if args.epochs < 1 or args.batch < 1 or not 0 < args.lr < 1:
        raise ValueError('positive epochs/batch and lr in (0,1) required')
    if not Path(args.foundation).is_dir():raise ValueError('local foundation required')
    torch.set_num_threads(8);torch.manual_seed(20260921)
    if args.train_workspace:
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    helper_path=Path(__file__).resolve().parents[2]/'examples/train_response_quality.py'
    spec=importlib.util.spec_from_file_location('quality_helpers',helper_path)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    manifest,splits=helper.load_data(args.data)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    model=Chatbot.from_foundation(args.foundation,revision=args.revision,local_files_only=True,max_input_tokens=512)
    model.to('cuda',dtype=torch.bfloat16).eval().requires_grad_(False)
    ids={word:model.tokenizer(word,add_special_tokens=False)['input_ids'] for word in ('yes','no')}
    if any(len(v)!=1 for v in ids.values()) or ids['yes']==ids['no']:
        raise ValueError('diagnostic requires distinct single-token yes/no labels')
    report={'role':'fixed-prompt development diagnostic; no training, calibration or promotion',
            'instructions':INSTRUCTIONS,'foundation':model.configuration()['foundation'],
            'foundation_asset_hashes':{p.name:helper.sha256(p) for p in sorted(Path(args.foundation).glob('*.safetensors'))},
            'script_sha256':helper.sha256(__file__),'data_manifest_sha256':helper.sha256(Path(args.data)/'manifest.json'),
            'label_ids':ids,'dtype':'bfloat16','max_tokens':512,'threshold':.5,
            'score_semantics':'first decoder-token probability conditioned on the yes/no alternatives, not calibrated correctness',
            'limitations':['Assistant review labels; known development data; foundation exposure unknown.',
                           'Authored instructions elicit inherited foundation behavior, not TensorCode learning.'], 'splits':{}}
    if args.train_workspace:
        report['role']='workspace-adaptation development diagnostic; no promotion'
        report['training']=train_workspace(model,splits['train'],helper,output,epochs=args.epochs,batch_size=args.batch,lr=args.lr)
        report['limitations'][1]='Authored instructions and assistant labels supervise only workspace, memory projection and gate; foundation frozen. No final questions.'
    with (output/'progress.jsonl').open('x') as stream:
        for split in ('calibration','development'):
            records=[];eligible=[];predictions=[];excluded=[];bypass_scores=[]
            for index,row in enumerate(splits[split]):
                receipt=assess(model,helper.model_inputs(row),yes_id=ids['yes'][0],no_id=ids['no'][0],
                               workspace_ablation=None if args.train_workspace else 'bypass')
                record={'id':row['id'],**receipt};records.append(record)
                if args.train_workspace and not receipt['input_truncated']:
                    bypass=assess(model,helper.model_inputs(row),yes_id=ids['yes'][0],no_id=ids['no'][0])
                    record['bypass_scores']=bypass['scores'];bypass_scores.append(bypass['scores'])
                if receipt['input_truncated']:excluded.append(row['id'])
                else:eligible.append(row);predictions.append(receipt['scores'])
                stream.write(json.dumps({'split':split,**record})+'\n');stream.flush()
                if (index+1)%20==0:print(json.dumps({'split':split,'completed':index+1}),flush=True)
            report['splits'][split]={'metrics':helper.metrics(eligible,predictions),'excluded':excluded,'records':records}
            if args.train_workspace:report['splits'][split]['same_foundation_bypass_metrics']=helper.metrics(eligible,bypass_scores)
    helper.write_json(output/'report.json',report)
    print(json.dumps(report['splits']['development']['metrics']),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--foundation',required=True);p.add_argument('--revision',required=True)
    p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    p.add_argument('--train-workspace',action='store_true')
    p.add_argument('--epochs',type=int,default=3);p.add_argument('--batch',type=int,default=4);p.add_argument('--lr',type=float,default=.001)
    run(p.parse_args())
