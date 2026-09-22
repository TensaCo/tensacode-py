"""Inference-only collapse diagnostic on four preselected reviewed calibration rows."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def dependencies():
    directory=Path(__file__).resolve().parent
    return (load('collapse_probe',directory/'probe_generative_quality.py'),
            load('collapse_helpers',directory.parents[1]/'examples/train_response_quality.py'))


def select_rows(model,rows):
    probe,helper=dependencies()
    groups={'all_known_good':[],'any_known_false':[]};overflow=[]
    for row in rows:
        targets=row.get('targets')
        if not isinstance(targets,dict) or set(targets)!=set(probe.INSTRUCTIONS) or any(value is not None and type(value) is not bool for value in targets.values()):
            raise ValueError('reviewed three-axis bool/None targets required')
        if any(len(model.tokenizer(prompt,truncation=False)['input_ids'])>model.config['max_input_tokens']
               for prompt in probe.prompts(helper.model_inputs(row)).values()):
            overflow.append(row['id']);continue
        group=('all_known_good' if all(value is True for value in targets.values()) else
               'any_known_false' if any(value is False for value in targets.values()) else None)
        if group is not None and len(groups[group])<2:groups[group].append(row)
    if any(len(group)!=2 for group in groups.values()):raise ValueError('selection requires two eligible known-good and two known-failure rows')
    return {'rows':[row for group in groups.values() for row in group],
            'selected_groups':{group:[row['id'] for row in selected] for group,selected in groups.items()},
            'overflow_ids':overflow}


def rms(value,mask=None):
    if mask is not None:value=value[mask.bool()]
    return float(value.detach().float().square().mean().sqrt())


def diagnose_row(model,row,*,autocast_dtype=None):
    import torch
    from torch.nn import functional as F
    probe,helper=dependencies()
    if model.training:raise ValueError('diagnostic requires eval mode')
    if model.config['memory_mode']!='contextualized_evidence':raise ValueError('diagnostic requires contextualized evidence memory')
    prompts=probe.prompts(helper.model_inputs(row))
    ids={word:model.tokenizer(word,add_special_tokens=False)['input_ids'] for word in ('yes','no')}
    if any(len(value)!=1 for value in ids.values()) or ids['yes']==ids['no']:raise ValueError('distinct single-token labels required')
    result={'id':row['id'],'axes':{}}
    with torch.no_grad(),probe.computation_context(model,autocast_dtype):
        for axis,prompt in prompts.items():
            count=len(model.tokenizer(prompt,truncation=False)['input_ids'])
            if count>model.config['max_input_tokens']:raise ValueError('selected row exceeds prompt budget')
            captured={}
            def encoder_hook(module,args,output):captured['tokens']=output.last_hidden_state
            def projection_hook(module,args,output):captured['update']=output
            handles=[model.foundation.get_encoder().register_forward_hook(encoder_hook),
                     model.memory_projection.register_forward_hook(projection_hook)]
            try:active=model.encode_workspace([prompt])
            finally:
                for handle in handles:handle.remove()
            tokens=captured['tokens'];update=captured['update'];mask=active['mask']
            bypass=dict(active,conditioning=tokens)
            token_rms=rms(tokens,mask);residual_rms=rms(active['conditioning']-tokens,mask)
            receipt={'input_token_count':count,'workspace':{'token_rms':token_rms,
                     'conditioning_rms':rms(active['conditioning'],mask),'update_rms':rms(update,mask),
                     'residual_rms':residual_rms,'residual_to_token_rms':residual_rms/token_rms if token_rms else None,
                     'memory_update':model.config['memory_update'],
                     'raw_gate':float(model.memory_gate.detach()),
                     'applied_gate':float(model.memory_gate.detach().float().tanh())}}
            gold=row['targets'][axis]
            target='yes' if gold is True else 'no' if gold is False else None
            if target is None:
                labels=torch.tensor([[ids['yes'][0]]],device=tokens.device)
            else:
                labels=model.tokenizer([target],padding=True,truncation=False,return_tensors='pt')['input_ids'].to(tokens.device)
                if labels.shape[1]>model.config['max_target_tokens']:raise ValueError('target would truncate')
                labels=labels.masked_fill(labels==model.tokenizer.pad_token_id,-100)
            for name,state in [('active',active),('bypass',bypass)]:
                decoded=model.decoder(dict(state,labels=labels))
                logits=decoded['logits'];first=logits[0,0].float()
                probabilities=torch.softmax(first,dim=-1)
                score=torch.softmax(first[[ids['no'][0],ids['yes'][0]]],dim=-1)[1]
                losses=F.cross_entropy(logits.float().reshape(-1,logits.shape[-1]),labels.reshape(-1),reduction='none').reshape(labels.shape)
                eos=labels==model.tokenizer.eos_token_id
                details={'conditional_yes':float(score),'yes_no_mass':float(probabilities[ids['yes'][0]]+probabilities[ids['no'][0]]),
                         'yes_probability':float(probabilities[ids['yes'][0]]),'no_probability':float(probabilities[ids['no'][0]]),
                         'gold_label':gold,'target_token_ids':labels[0].tolist() if target else None,
                         'token_cross_entropy':losses[0].tolist() if target else None,
                         'eos_cross_entropy':float(losses[eos].mean()) if target and eos.any() else None,
                         'sequence_loss':float(decoded['loss']) if target else None}
                if name=='bypass':
                    batch=model.tokenizer([prompt],padding=True,truncation=True,max_length=model.config['max_input_tokens'],return_tensors='pt')
                    batch={key:value.to(tokens.device) for key,value in batch.items() if key in ('input_ids','attention_mask')}
                    native=model.foundation(**batch,labels=labels,return_dict=True,use_cache=False)
                    details.update(native_logits_exact=torch.equal(native.logits,logits),
                                   native_loss_exact=torch.equal(native.loss,decoded['loss']),
                                   native_logits_max_abs_difference=float((native.logits.float()-logits.float()).abs().max()))
                receipt[name]=details
            result['axes'][axis]=receipt
    return result


def run(args):
    directory=Path(__file__).resolve().parent
    verifier=load('collapse_verifier',directory/'verify_generative_quality_reload.py')
    verifier.configure_runtime(args.device)
    import torch
    from tensorcode.tools.chatbot import Chatbot
    probe,helper=dependencies()
    gate=load('collapse_gate',directory/'assess_quality_gate.py')
    run_path=Path(args.run);data=Path(args.data);output=Path(args.output)
    if output.exists():raise FileExistsError(output)
    report=json.loads((run_path/'report.json').read_text())
    _,splits=helper.load_data(data)
    gate.assess_report(report,splits['calibration'],manifest_sha256=verifier.sha256(data/'manifest.json'),split='calibration')
    if report.get('instructions')!=probe.INSTRUCTIONS:raise ValueError('recorded instructions differ')
    if report.get('dtype')!='float32':raise ValueError('diagnostic requires saved float32 master weights')
    if report.get('autocast_dtype') not in (None,'bfloat16'):raise ValueError('unsupported report autocast')
    model=Chatbot.from_pretrained(run_path/'model',device=args.device).eval()
    verifier.validate_conditioning(report,model)
    if any(parameter.dtype!=torch.float32 for parameter in model.parameters()):raise ValueError('artifact must retain float32 weights')
    if report.get('foundation')!=model.configuration().get('foundation'):raise ValueError('foundation provenance differs')
    if report.get('max_tokens')!=model.config['max_input_tokens']:raise ValueError('input budget differs')
    ids={word:model.tokenizer(word,add_special_tokens=False)['input_ids'] for word in ('yes','no')}
    if report.get('label_ids')!=ids:raise ValueError('recorded label IDs differ')
    trained='training' in report
    if trained and not isinstance(report['training'],dict):raise ValueError('training metadata must be an object')
    if trained:
        recorded=report['training'].get('foundation_sha256_after')
        if recorded is None and report['training'].get('foundation_unchanged') is True:
            recorded=report['training'].get('foundation_sha256')
        if recorded is not None and helper.tensor_digest(model.foundation)!=recorded:raise ValueError('recorded foundation digest differs')
    selection=select_rows(model,splits['calibration'])
    expected={record['id']:record for record in report['splits']['calibration']['records']}
    # Verify selected saved receipts in their original three-prompt batches before
    # changing inference precision or decomposing each axis into a separate call.
    for row in selection['rows']:
        record=expected[row['id']]
        if trained != ('bypass_scores' in record):raise ValueError('recorded training/bypass mode differs')
        for mode in ([None,'bypass'] if trained else ['bypass']):
            actual=probe.assess(model,helper.model_inputs(row),yes_id=ids['yes'][0],no_id=ids['no'][0],workspace_ablation=mode,
                                autocast_dtype=torch.bfloat16 if report.get('autocast_dtype')=='bfloat16' else None)
            wanted_scores=record['bypass_scores'] if trained and mode=='bypass' else record['scores']
            if actual['scores']!=wanted_scores or any(actual[key]!=record[key] for key in ('input_token_counts','input_truncated')):
                raise AssertionError('selected saved receipt differs after reload')
    records={}
    for name,dtype in [('float32',None),('bfloat16_autocast',torch.bfloat16)]:
        records[name]=[diagnose_row(model,row,autocast_dtype=dtype) for row in selection['rows']]
    result={'role':'inference-only optimization-collapse diagnostic; no qualification or fitting',
            'comparison_scope':'same saved float32 artifact in two computation precisions; no original-foundation comparison',
            'selection_rule':'first two all-known-good and first two any-known-false calibration rows fitting all three prompts; prepared order',
            'selected_groups':selection['selected_groups'],'overflow_ids':selection['overflow_ids'],
            'selected_saved_receipts_exact':True,'records':records,
            'report_sha256':verifier.sha256(run_path/'report.json'),'data_manifest_sha256':verifier.sha256(data/'manifest.json'),
            'artifact_sha256':{name:verifier.sha256(run_path/'model'/name) for name in ('tensorcode_config.json','model.safetensors')},
            'script_sha256':verifier.sha256(__file__),'probe_script_sha256':verifier.sha256(probe.__file__),
            'limitations':['Reviewed labels select diagnostic examples; no training, threshold fitting, or final questions.',
                           'Each diagnostic axis runs separately; saved receipt validation uses the original three-axis batch.',
                           'Hashes identify current files; this small diagnostic cannot establish generalization.',
                           'Unknown axes have no gold-target loss. Absolute probability mass differs from conditional yes/no scores.']}
    with output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'selected':selection['selected_groups'],'saved_receipts_exact':True}),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True);parser.add_argument('--data',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    run(parser.parse_args())
