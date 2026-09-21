import importlib.util
import json
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest
import torch


def module(name):
    path=Path(__file__).parents[2]/f'.development/experiments/{name}.py'
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


@pytest.fixture
def prepared(tmp_path):
    from tensorcode.tools.chatbot import Chatbot
    verifier=module('verify_generative_quality_reload')
    previous_threads=torch.get_num_threads()
    verifier.configure_runtime('cpu')
    config=runpy.run_path(str(Path(__file__).parents[1]/'models/test_chatbot_model.py'))['tiny_config']()
    config['max_input_tokens']=256
    vocabulary=json.loads(config['tokenizer_json']);vocabulary['model']['vocab'].update(yes=8,no=9)
    config['tokenizer_json']=json.dumps(vocabulary);config['foundation_config']['vocab_size']=10
    data=tmp_path/'data';data.mkdir()
    rows={}
    for split in ('train','calibration','development'):
        rows[split]=[{'id':split,'question_id':split,'question':'hello','candidate':'world',
                      'evidence':[{'id':split,'source_id':split,'text':'hello world '+split}],
                      'targets':dict.fromkeys(('support','completeness','constraints'),True)}]
        (data/f'{split}.jsonl').write_text(json.dumps(rows[split][0])+'\n')
    (data/'manifest.json').write_text(json.dumps({'files':{p.name:verifier.sha256(p) for p in data.glob('*.jsonl')}}))
    run=tmp_path/'run';run.mkdir()
    yield verifier,Chatbot(config).eval(),rows,SimpleNamespace(run=run,data=data,output=tmp_path/'verification.json',device='cpu')
    torch.set_num_threads(previous_threads)


def write_report(model,rows,args,*,autocast_dtype=None,training=True):
    probe=module('probe_generative_quality')
    verifier=module('verify_generative_quality_reload')
    report={'data_manifest_sha256':verifier.sha256(args.data/'manifest.json'),
            'instructions':probe.INSTRUCTIONS,'foundation':model.configuration().get('foundation'),'label_ids':{'yes':[8],'no':[9]},
            'max_tokens':model.config['max_input_tokens'],'dtype':str(next(model.parameters()).dtype).removeprefix('torch.'),
            'autocast_dtype':'bfloat16' if autocast_dtype else None,'splits':{}}
    if training:report['training']={'epochs':1}
    helper=verifier.load_module('quality_helper',Path(__file__).parents[2]/'examples/train_response_quality.py')
    for split in ('calibration','development'):
        row=rows[split][0];inputs=helper.model_inputs(row)
        result=probe.assess(model,inputs,yes_id=8,no_id=9,workspace_ablation=None if training else 'bypass',autocast_dtype=autocast_dtype)
        record={'id':row['id'],**result}
        if training:record['bypass_scores']=probe.assess(model,inputs,yes_id=8,no_id=9,autocast_dtype=autocast_dtype)['scores']
        report['splits'][split]={'records':[record],'excluded':[row['id']] if result['input_truncated'] else []}
    model.save_pretrained(args.run/'model')
    (args.run/'report.json').write_text(json.dumps(report))
    return report


@pytest.mark.parametrize('dtype,autocast_dtype,training',[(torch.float32,None,False),(torch.float32,torch.bfloat16,True),(torch.bfloat16,None,True)])
def test_owned_model_reload_exact_scores_preserves_dtype(prepared,dtype,autocast_dtype,training):
    verifier,model,rows,args=prepared
    write_report(model.to(dtype=dtype),rows,args,autocast_dtype=autocast_dtype,training=training)
    result=verifier.run(args)
    assert result['records']==2 and result['all_receipts_exact']
    assert result['bypass_records']==(2 if training else 0)
    assert result['parameter_dtype']==str(dtype).removeprefix('torch.')
    assert set(result['artifact_sha256'])=={'tensorcode_config.json','model.safetensors'}
    with pytest.raises(FileExistsError):verifier.run(args)


@pytest.mark.parametrize('mutation',[
    lambda r:r.update(data_manifest_sha256='wrong'),
    lambda r:r['instructions'].update(support='changed'),
    lambda r:r['label_ids'].update(yes=[9]),
    lambda r:r.update(max_tokens=2),
    lambda r:r.update(dtype='float64'),
    lambda r:r.update(dtype='bfloat16'),
    lambda r:r.update(autocast_dtype='float16'),
    lambda r:r['splits']['development']['records'][0].update(id='different'),
    lambda r:r['splits']['development']['records'][0].update(input_truncated=True),
    lambda r:r['splits']['development']['records'].append(dict(r['splits']['development']['records'][0])),
    lambda r:r['splits']['development']['records'][0]['scores'].update(support=.123456789),
    lambda r:r['splits']['development']['records'][0]['input_token_counts'].update(support=0),
    lambda r:r['splits']['development']['records'][0]['bypass_scores'].update(support=.123456789),
])
def test_mutated_report_rejected_without_success_output(prepared,mutation):
    verifier,model,rows,args=prepared
    report=write_report(model,rows,args)
    mutation(report);(args.run/'report.json').write_text(json.dumps(report))
    with pytest.raises((ValueError,AssertionError)):verifier.run(args)
    assert not args.output.exists()


@pytest.mark.parametrize('omit_bypass',[False,True])
def test_truncated_receipts_are_recomputed_and_counted(prepared,omit_bypass):
    verifier,model,rows,args=prepared
    model.config['max_input_tokens']=2
    report=write_report(model,rows,args)
    if omit_bypass:
        for split in report['splits'].values():
            for record in split['records']:record.pop('bypass_scores')
        (args.run/'report.json').write_text(json.dumps(report))
    result=verifier.run(args)
    assert result['records']==2 and result['bypass_records']==0
    assert result['all_receipts_exact']


def test_modified_prepared_data_rejected(prepared):
    verifier,model,rows,args=prepared
    write_report(model,rows,args)
    with (args.data/'development.jsonl').open('a') as stream:stream.write('\n')
    with pytest.raises(ValueError,match='checksum'):verifier.run(args)
    assert not args.output.exists()


def test_cli_recomputes_in_fresh_python_process(prepared):
    import subprocess
    import sys
    verifier,model,rows,args=prepared
    write_report(model,rows,args,autocast_dtype=torch.bfloat16)
    command=[sys.executable,verifier.__file__,'--run',str(args.run),'--data',str(args.data),
             '--output',str(args.output),'--device','cpu']
    result=subprocess.run(command,check=False,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    receipt=json.loads(args.output.read_text())
    assert receipt['fresh_process'] and receipt['all_receipts_exact']
    assert receipt['records']==2 and receipt['bypass_records']==2


def test_trained_report_cannot_drop_all_bypass_fields(prepared):
    verifier,model,rows,args=prepared
    report=write_report(model,rows,args,training=True)
    for split in report['splits'].values():
        for record in split['records']:record.pop('bypass_scores',None)
    (args.run/'report.json').write_text(json.dumps(report))
    with pytest.raises(ValueError,match='trained.*bypass'):verifier.run(args)
    assert not args.output.exists()


def test_frozen_report_rejects_unexpected_bypass_fields(prepared):
    verifier,model,rows,args=prepared
    report=write_report(model,rows,args,training=True)
    del report['training']
    (args.run/'report.json').write_text(json.dumps(report))
    with pytest.raises(ValueError,match='frozen.*bypass'):verifier.run(args)
    assert not args.output.exists()


def test_training_metadata_must_be_object(prepared):
    verifier,model,rows,args=prepared
    report=write_report(model,rows,args,training=True)
    report['training']=False
    (args.run/'report.json').write_text(json.dumps(report))
    with pytest.raises(ValueError,match='training.*object'):verifier.run(args)
    assert not args.output.exists()


@pytest.mark.parametrize('legacy',[False,True])
def test_recorded_foundation_digest_is_verified(prepared,legacy):
    verifier,model,rows,args=prepared
    report=write_report(model,rows,args)
    helper=verifier.load_module('digest_helper',Path(__file__).parents[2]/'examples/train_response_quality.py')
    digest=helper.tensor_digest(model.foundation)
    if legacy:report['training'].update(foundation_unchanged=True,foundation_sha256=digest)
    else:report['training']['foundation_sha256_after']=digest
    (args.run/'report.json').write_text(json.dumps(report))
    result=verifier.run(args)
    assert result['recorded_foundation_digest_verified']
    assert any('current files' in text for text in result['limitations'])


@pytest.mark.parametrize('legacy',[False,True])
def test_mismatched_recorded_foundation_digest_rejected(prepared,legacy):
    verifier,model,rows,args=prepared
    report=write_report(model,rows,args)
    if legacy:report['training'].update(foundation_unchanged=True,foundation_sha256='wrong')
    else:report['training']['foundation_sha256_after']='wrong'
    (args.run/'report.json').write_text(json.dumps(report))
    with pytest.raises(ValueError,match='foundation digest'):verifier.run(args)
    assert not args.output.exists()


def test_mismatched_foundation_provenance_rejected(prepared):
    verifier,model,rows,args=prepared
    report=write_report(model,rows,args)
    report['foundation']={'repo':'different','revision':'wrong'}
    (args.run/'report.json').write_text(json.dumps(report))
    with pytest.raises(ValueError,match='foundation provenance'):verifier.run(args)
    assert not args.output.exists()
