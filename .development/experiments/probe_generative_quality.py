"""Frozen or adapted quality judgments on reviewed development data."""
from __future__ import annotations
import argparse
from contextlib import nullcontext
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


def computation_context(model, dtype=None):
    import torch
    return nullcontext() if dtype is None else torch.autocast(next(model.parameters()).device.type,dtype=dtype)


def state_digest(value):
    """Hash exact nested checkpoint values without retaining model-sized copies."""
    import torch
    digest=hashlib.sha256()
    def token(value):
        data=json.dumps(value,ensure_ascii=False,allow_nan=False).encode()
        digest.update(len(data).to_bytes(8,'big'));digest.update(data)
    def visit(value):
        if isinstance(value,torch.Tensor):
            token(['tensor',str(value.dtype),list(value.shape)])
            flat=value.detach().reshape(-1)
            # Bound each device-to-host copy to one million elements.
            for start in range(0,flat.numel(),1_000_000):
                chunk=flat[start:start+1_000_000].to('cpu').contiguous()
                digest.update(chunk.view(torch.uint8).numpy().tobytes())
        elif isinstance(value,dict):
            token(['dict',len(value)])
            for key in sorted(value,key=lambda k:(type(k).__name__,repr(k))):
                visit(key);visit(value[key])
        elif isinstance(value,(list,tuple)):
            token([type(value).__name__,len(value)])
            for item in value:visit(item)
        elif value is None or type(value) in (str,int,float,bool):
            token([type(value).__name__,value])
        else:raise TypeError(f'Unsupported digest value: {type(value).__name__}')
    visit(value)
    return digest.hexdigest()


def assess(model,row,*,yes_id,no_id,workspace_ablation='bypass',autocast_dtype=None):
    import torch
    values=prompts(row)
    lengths={axis:len(model.tokenizer(text,truncation=False)['input_ids']) for axis,text in values.items()}
    truncated=any(n>model.config['max_input_tokens'] for n in lengths.values())
    result={'input_token_counts':lengths,'input_truncated':truncated,'scores':None}
    if truncated:
        return result
    with torch.no_grad(), computation_context(model,autocast_dtype):
        # Active and native-memory-bypass receipts use identical computation settings.
        state=model.encode_workspace(list(values.values()),workspace_ablation=workspace_ablation)
        labels=torch.full((len(values),1),yes_id,dtype=torch.long,device=state['conditioning'].device)
        logits=model.decoder(dict(state,labels=labels))['logits'][:,0,:]
        scores=torch.softmax(logits[:,[no_id,yes_id]].float(),dim=-1)[:,1].tolist()
    result['scores']=dict(zip(values,scores))
    return result


def workspace_trainer(model, *, lr):
    import torch
    from tensorcode.training import Trainer
    model.foundation.requires_grad_(False)
    trainer=Trainer.from_tool(model,optimizer=lambda parameters:torch.optim.AdamW(parameters,lr=lr))
    # Deterministic frozen foundation and adapter computation; eval does not disable gradients.
    model.eval()
    return trainer


def train_workspace(model, rows, helper, output, *, epochs, batch_size, lr,
                    train_foundation=False, foundation_lr=2e-5, autocast_dtype=None):
    import gc
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
    # The diagnostic loader froze all parameters; activate the requested parameter groups.
    model.workspace.requires_grad_(True);model.memory_projection.requires_grad_(True)
    model.memory_gate.requires_grad_(True)
    def make_trainer():
        if not train_foundation:
            return workspace_trainer(model,lr=lr)
        from tensorcode.training import Trainer
        if any(p.dtype!=torch.float32 for p in model.parameters()):
            raise ValueError('foundation adaptation requires float32 master parameters')
        model.foundation.requires_grad_(True)
        foundation_ids={id(p) for p in model.foundation.parameters()}
        def optimizer(parameters):
            return torch.optim.AdamW([
                {'params':[p for p in parameters if id(p) in foundation_ids],'lr':foundation_lr},
                {'params':[p for p in parameters if id(p) not in foundation_ids],'lr':lr}],foreach=False)
        result=Trainer.from_tool(model,optimizer=optimizer)
        model.eval()
        return result
    trainer=make_trainer()
    foundation_before=helper.tensor_digest(model.foundation)
    adapter_ids={id(p) for module in (model.workspace,model.memory_projection) for p in module.parameters()}|{id(model.memory_gate)}
    adapter_before={name:state_digest(value) for name,value in model.named_parameters() if id(value) in adapter_ids}
    def captured_step(batch,source):
        with computation_context(model,autocast_dtype):
            # Replay rebuilds the differentiable graph; avoid retaining a second
            # 3B-model activation graph merely to capture the supervised calls.
            with torch.no_grad() if train_foundation else nullcontext():
                experience=trainer.capture([r[2] for r in batch],[r[3] for r in batch],source=source)
            return trainer.step(experience)
    rng=random.Random(20260923);losses=[]
    for epoch in range(epochs):
        order=list(pairs);rng.shuffle(order);epoch_losses=[]
        for start in range(0,len(order),batch_size):
            batch=order[start:start+batch_size]
            epoch_losses.append(captured_step(batch,'assistant-reviewed response-quality-v2'))
        losses.append(sum(epoch_losses)/len(epoch_losses))
        print(json.dumps({'epoch':epoch+1,'loss':losses[-1],'steps':trainer.steps}),flush=True)
    trainer.save_checkpoint(output/'training',progress={'epochs':epochs,'supervised_axis_examples':len(pairs)})
    probe=pairs[:batch_size]
    def step():
        return captured_step(probe,'continuation probe; restored afterwards')
    expected_loss=step()
    expected=state_digest(model.state_dict())
    expected_optimizer=state_digest(trainer.optimizer.state_dict())
    trainer.optimizer.zero_grad(set_to_none=True)
    # Fresh optimizer avoids copying its old moment buffers for load rollback.
    del trainer
    gc.collect()
    if next(model.parameters()).device.type=='cuda':torch.cuda.empty_cache()
    trainer=make_trainer()
    trainer.load_checkpoint(output/'training')
    actual_loss=step()
    exact=(actual_loss==expected_loss and expected==state_digest(model.state_dict())
           and expected_optimizer==state_digest(trainer.optimizer.state_dict()))
    if not exact:raise AssertionError('optimizer continuation differed')
    trainer.optimizer.zero_grad(set_to_none=True)
    # Fresh optimizer avoids copying its old moment buffers for load rollback.
    del trainer
    gc.collect()
    if next(model.parameters()).device.type=='cuda':torch.cuda.empty_cache()
    trainer=make_trainer()
    trainer.load_checkpoint(output/'training')
    model.eval();model.save_pretrained(output/'model')
    foundation_after=helper.tensor_digest(model.foundation)
    unchanged=foundation_before==foundation_after
    if not train_foundation and not unchanged:raise AssertionError('frozen foundation changed')
    if train_foundation and unchanged:raise AssertionError('native foundation did not change')
    return {'epochs':epochs,'batch':batch_size,'lr':lr,'seed':20260923,'loss':'native teacher-forced yes/no sequence cross entropy',
            'supervised_axis_examples':len(pairs),'excluded':excluded,'epoch_losses':losses,'steps':trainer.steps,
            'foundation_unchanged':unchanged,'foundation_sha256':foundation_before,
            'foundation_sha256_before':foundation_before,'foundation_sha256_after':foundation_after,
            'foundation_lr':foundation_lr if train_foundation else None,
            'autocast_dtype':str(autocast_dtype),'optimizer_continuation_exact':exact,
            'continuation_scope':'one fixed next batch; epoch shuffle RNG is not checkpointed',
            'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'adapter_parameters_changed':sum(adapter_before[n]!=state_digest(v) for n,v in model.named_parameters() if n in adapter_before)}


def evaluate_splits(model,splits,helper,ids,stream,*,compare_workspace=False,autocast_dtype=None):
    result={}
    for split in ('calibration','development'):
        records=[];eligible=[];predictions=[];excluded=[];bypass_scores=[]
        for index,row in enumerate(splits[split]):
            receipt=assess(model,helper.model_inputs(row),yes_id=ids['yes'][0],no_id=ids['no'][0],
                           workspace_ablation=None if compare_workspace else 'bypass',autocast_dtype=autocast_dtype)
            record={'id':row['id'],**receipt};records.append(record)
            if compare_workspace and not receipt['input_truncated']:
                bypass=assess(model,helper.model_inputs(row),yes_id=ids['yes'][0],no_id=ids['no'][0],autocast_dtype=autocast_dtype)
                record['bypass_scores']=bypass['scores'];bypass_scores.append(bypass['scores'])
            if receipt['input_truncated']:excluded.append(row['id'])
            else:eligible.append(row);predictions.append(receipt['scores'])
            stream.write(json.dumps({'split':split,**record})+'\n');stream.flush()
            if (index+1)%20==0:print(json.dumps({'split':split,'completed':index+1}),flush=True)
        result[split]={'metrics':helper.metrics(eligible,predictions),'excluded':excluded,'records':records}
        if compare_workspace:result[split]['same_foundation_bypass_metrics']=helper.metrics(eligible,bypass_scores)
    return result


def run(args):
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    from tensorcode.tools.chatbot import Chatbot
    if not torch.cuda.is_available():raise RuntimeError('real models require authorized CUDA host')
    if args.epochs < 1 or args.batch < 1 or not 0 < args.lr < 1 or not 0 < args.foundation_lr < 1:
        raise ValueError('positive epochs/batch and lr in (0,1) required')
    if not Path(args.foundation).is_dir():raise ValueError('local foundation required')
    torch.set_num_threads(8);torch.manual_seed(20260921)
    training=args.train_workspace or args.train_foundation
    if args.train_workspace and args.train_foundation:raise ValueError('training modes are mutually exclusive')
    autocast_dtype=torch.bfloat16 if args.train_foundation else None
    if training:
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
    model.to('cuda',dtype=torch.float32 if args.train_foundation else torch.bfloat16).eval().requires_grad_(False)
    if args.train_foundation:torch.cuda.reset_peak_memory_stats()
    ids={word:model.tokenizer(word,add_special_tokens=False)['input_ids'] for word in ('yes','no')}
    if any(len(v)!=1 for v in ids.values()) or ids['yes']==ids['no']:
        raise ValueError('diagnostic requires distinct single-token yes/no labels')
    report={'role':'fixed-prompt development diagnostic; no training, calibration or promotion',
            'instructions':INSTRUCTIONS,'foundation':model.configuration()['foundation'],
            'memory_update':model.config['memory_update'],'memory_mode':model.config['memory_mode'],
            'workspace':model.config['workspace'],
            'foundation_asset_hashes':{p.name:helper.sha256(p) for p in sorted(Path(args.foundation).glob('*.safetensors'))},
            'script_sha256':helper.sha256(__file__),'data_manifest_sha256':helper.sha256(Path(args.data)/'manifest.json'),
            'label_ids':ids,'dtype':'float32' if args.train_foundation else 'bfloat16',
            'autocast_dtype':'bfloat16' if args.train_foundation else None,'max_tokens':512,'threshold':.5,
            'score_semantics':'first decoder-token probability conditioned on the yes/no alternatives, not calibrated correctness',
            'limitations':['Assistant review labels; known development data; foundation exposure unknown.',
                           'Authored instructions elicit inherited foundation behavior, not TensorCode learning.'], 'splits':{}}
    if training:
        with (output/'before-progress.jsonl').open('x') as stream:
            report['before_training']=evaluate_splits(model,splits,helper,ids,stream,
                                                       compare_workspace=True,autocast_dtype=autocast_dtype)
        helper.write_json(output/'before-training.json',report)
    if training:
        report['role']='foundation-adaptation development diagnostic; no promotion' if args.train_foundation else 'workspace-adaptation development diagnostic; no promotion'
        report['training']=train_workspace(model,splits['train'],helper,output,epochs=args.epochs,batch_size=args.batch,lr=args.lr,
                                           train_foundation=args.train_foundation,foundation_lr=args.foundation_lr,autocast_dtype=autocast_dtype)
        report['limitations'][1]=('Authored instructions and assistant labels adapt the inherited foundation and workspace; no final questions or qualification.'
                                  if args.train_foundation else 'Authored instructions and assistant labels supervise only workspace, memory projection and gate; foundation frozen. No final questions.')
        if args.train_foundation:report['limitations'].append('Checkpoint restore still holds loaded tensors and a model rollback copy; peak CUDA counters omit CPU/unified-memory pressure.')
    with (output/'progress.jsonl').open('x') as stream:
        report['splits']=evaluate_splits(model,splits,helper,ids,stream,
                                        compare_workspace=training,autocast_dtype=autocast_dtype)
    if args.train_foundation:
        report['cuda_memory']={'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                               'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
    helper.write_json(output/'report.json',report)
    print(json.dumps(report['splits']['development']['metrics']),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--foundation',required=True);p.add_argument('--revision',required=True)
    p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    modes=p.add_mutually_exclusive_group()
    modes.add_argument('--train-workspace',action='store_true')
    modes.add_argument('--train-foundation',action='store_true')
    p.add_argument('--foundation-lr',type=float,default=2e-5)
    p.add_argument('--epochs',type=int,default=3);p.add_argument('--batch',type=int,default=4);p.add_argument('--lr',type=float,default=.001)
    run(p.parse_args())
