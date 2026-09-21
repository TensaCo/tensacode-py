"""Recompute saved generative quality receipts exactly after owned-artifact reload.

Use CUDA on the authorized training host for real models. CPU supports tiny
mechanism fixtures. This verifies persistence, not model quality or qualification.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def configure_runtime(device):
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import torch
    if device not in ('cpu','cuda'):raise ValueError('device must be cpu or cuda')
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA host required for real model verification')
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def run(args,*,fresh_process=False):
    configure_runtime(args.device)
    import torch
    from tensorcode.tools.chatbot import Chatbot
    run_path=Path(args.run);data_path=Path(args.data);output=Path(args.output)
    if output.exists():raise FileExistsError(output)
    root=Path(__file__).resolve().parents[2]
    helper=load_module('quality_reload_helpers',root/'examples/train_response_quality.py')
    probe=load_module('quality_reload_probe',Path(__file__).with_name('probe_generative_quality.py'))
    gate=load_module('quality_reload_validation',Path(__file__).with_name('assess_quality_gate.py'))
    report_path=run_path/'report.json'
    report=json.loads(report_path.read_text())
    _,splits=helper.load_data(data_path)
    if report.get('data_manifest_sha256')!=sha256(data_path/'manifest.json'):
        raise ValueError('report data manifest checksum mismatch')
    if report.get('instructions')!=probe.INSTRUCTIONS:raise ValueError('quality instructions differ')
    dtype=report.get('dtype')
    if dtype not in ('float32','bfloat16'):raise ValueError('unsupported report parameter dtype')
    autocast=report.get('autocast_dtype')
    if autocast not in (None,'bfloat16'):raise ValueError('unsupported report autocast dtype')
    autocast_dtype=torch.bfloat16 if autocast=='bfloat16' else None
    trained='training' in report
    if trained and not isinstance(report['training'],dict):raise ValueError('training metadata must be an object')
    expected_splits=report.get('splits')
    if not isinstance(expected_splits,dict) or not {'calibration','development'}<=set(expected_splits):
        raise ValueError('report requires calibration and development records')
    for split in ('calibration','development'):
        gate.validate_block(expected_splits[split],splits[split])
        records=expected_splits[split]['records']
        if [row['id'] for row in splits[split]]!=[record['id'] for record in records]:
            raise ValueError('report record order differs from prepared data')
        for record in records:
            if trained and not record['input_truncated'] and 'bypass_scores' not in record:
                raise ValueError('trained report requires bypass scores for every eligible record')
            if not trained and 'bypass_scores' in record:
                raise ValueError('frozen report cannot contain bypass comparison fields')
            counts=record.get('input_token_counts')
            if not isinstance(counts,dict) or set(counts)!=set(probe.INSTRUCTIONS) or any(type(value) is not int or value<0 for value in counts.values()):
                raise ValueError('exact integer input counts required for every axis')
    model=Chatbot.from_pretrained(run_path/'model',device=args.device).eval()
    if report.get('foundation')!=model.configuration().get('foundation'):
        raise ValueError('artifact foundation provenance differs from report')
    recorded_digest=None
    if trained:
        training=report['training']
        if 'foundation_sha256_after' in training:
            recorded_digest=training['foundation_sha256_after']
            if not isinstance(recorded_digest,str) or not recorded_digest:
                raise ValueError('recorded foundation digest must be a nonempty string')
        elif training.get('foundation_unchanged') is True and 'foundation_sha256' in training:
            recorded_digest=training['foundation_sha256']
            if not isinstance(recorded_digest,str) or not recorded_digest:
                raise ValueError('recorded foundation digest must be a nonempty string')
    if recorded_digest is not None and helper.tensor_digest(model.foundation)!=recorded_digest:
        raise ValueError('artifact foundation digest differs from report')
    dtypes={str(parameter.dtype).removeprefix('torch.') for parameter in model.parameters()}
    if dtypes!={dtype}:raise ValueError('artifact parameter dtype differs from report')
    if type(report.get('max_tokens')) is not int or model.config['max_input_tokens']!=report['max_tokens']:
        raise ValueError('artifact input token budget differs from report')
    ids={word:model.tokenizer(word,add_special_tokens=False)['input_ids'] for word in ('yes','no')}
    if (any(len(value)!=1 for value in ids.values()) or ids['yes']==ids['no']
            or report.get('label_ids')!=ids):raise ValueError('native yes/no label IDs differ from report')
    counts={};bypass_count=0
    for split in ('calibration','development'):
        count=0
        for row,expected in zip(splits[split],expected_splits[split]['records'],strict=True):
            actual=probe.assess(model,helper.model_inputs(row),yes_id=ids['yes'][0],no_id=ids['no'][0],
                                workspace_ablation=None if trained else 'bypass',autocast_dtype=autocast_dtype)
            for key,value in actual.items():
                if value!=expected[key]:raise AssertionError(f'{split}/{row["id"]}: {key} differs after reload')
            if trained and not actual['input_truncated']:
                bypass=probe.assess(model,helper.model_inputs(row),yes_id=ids['yes'][0],no_id=ids['no'][0],autocast_dtype=autocast_dtype)
                if bypass['scores']!=expected['bypass_scores']:raise AssertionError(f'{split}/{row["id"]}: bypass scores differ after reload')
                # Both paths tokenize identical prompts; verify complete recomputed
                # receipts rather than assuming the bypass changes only scores.
                if any(bypass[key]!=expected[key] for key in ('input_token_counts','input_truncated')):
                    raise AssertionError(f'{split}/{row["id"]}: bypass input receipt differs after reload')
                bypass_count+=1
            count+=1
        counts[split]=count
        print(json.dumps({'split':split,'exact_records':count}),flush=True)
    result={'role':'exact saved-artifact receipt verification; not quality qualification',
            'fresh_process':fresh_process,'records':sum(counts.values()),'split_records':counts,
            'bypass_records':bypass_count,'all_receipts_exact':True,'all_present_bypass_receipts_exact':True,
            'parameter_dtype':dtype,'autocast_dtype':autocast,'device':args.device,
            'recorded_foundation_digest_verified':recorded_digest is not None,
            'limitations':['Hashes identify current files verified by this run, not independently established original artifact identity.',
                           'Exact receipt equivalence checks recorded behavior; a recorded foundation digest is also checked when available.'],
            'report_sha256':sha256(report_path),'data_manifest_sha256':sha256(data_path/'manifest.json'),
            'data_files_sha256':{f'{split}.jsonl':sha256(data_path/f'{split}.jsonl') for split in splits},
            'artifact_sha256':{name:sha256(run_path/'model'/name) for name in ('tensorcode_config.json','model.safetensors')},
            'verification_script_sha256':sha256(__file__),'probe_script_sha256':sha256(probe.__file__),
            'scope':'Every saved calibration/development receipt, including truncation; exact equality, no tolerance or final questions.'}
    with output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(result),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    run(parser.parse_args(),fresh_process=True)
