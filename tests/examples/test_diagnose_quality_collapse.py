import importlib.util
import json
from pathlib import Path
import runpy

import pytest
import torch


def runner():
    path=Path(__file__).parents[2]/'.development/experiments/diagnose_quality_collapse.py'
    spec=importlib.util.spec_from_file_location('collapse_diagnostic',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def model_fixture():
    from tokenizers import Tokenizer,processors
    from tensorcode.tools.chatbot import Chatbot
    config=runpy.run_path(str(Path(__file__).parents[1]/'models/test_chatbot_model.py'))['tiny_config']()
    vocab=json.loads(config['tokenizer_json']);vocab['model']['vocab'].update(yes=8,no=9)
    tokenizer=Tokenizer.from_str(json.dumps(vocab))
    tokenizer.post_processor=processors.TemplateProcessing(single='$A </s>',special_tokens=[('</s>',1)])
    config['tokenizer_json']=tokenizer.to_str();config['foundation_config']['vocab_size']=10
    config['max_input_tokens']=256
    return Chatbot(config).eval()


def rows_fixture():
    return [{'id':str(i),'question':'hello','candidate':'world',
             'evidence':[{'id':'a','text':'hello world'}],
             'targets':{'support':True if i<3 else False,'completeness':True,'constraints':True if i<3 else None}}
            for i in range(6)]


def test_selection_is_fixed_first_two_good_and_failure_and_excludes_overflow():
    model=model_fixture();rows=rows_fixture()
    rows[0]['question']=' '.join(['hello']*300)
    result=runner().select_rows(model,rows)
    assert [row['id'] for row in result['rows']]==['1','2','3','4']
    assert result['overflow_ids']==['0']
    assert result['selected_groups']=={'all_known_good':['1','2'],'any_known_false':['3','4']}


@pytest.mark.parametrize('autocast_dtype',[None,torch.bfloat16])
def test_native_parity_token_losses_and_conditioning_diagnostic(autocast_dtype):
    model=model_fixture();mod=runner()
    before={name:value.clone() for name,value in model.state_dict().items()}
    result=mod.diagnose_row(model,rows_fixture()[0],autocast_dtype=autocast_dtype)
    assert set(result['axes'])=={'support','completeness','constraints'}
    for receipt in result['axes'].values():
        assert receipt['bypass']['native_logits_exact']
        assert receipt['bypass']['native_loss_exact']
        for mode in ('active','bypass'):
            assert 0<receipt[mode]['yes_no_mass']<=1
            assert 0<=receipt[mode]['conditional_yes']<=1
            assert receipt[mode]['target_token_ids']==[8,1]
            assert len(receipt[mode]['token_cross_entropy'])==2
            assert receipt[mode]['eos_cross_entropy'] is not None
        assert receipt['workspace']['token_rms']>0
        assert receipt['workspace']['conditioning_rms']>0
        assert receipt['workspace']['residual_rms']>=0
    assert all(torch.equal(value,model.state_dict()[name]) for name,value in before.items())
    assert all(parameter.grad is None for parameter in model.parameters())


def test_unknown_axis_has_no_fabricated_target_loss():
    result=runner().diagnose_row(model_fixture(),rows_fixture()[-1])
    for mode in ('active','bypass'):
        receipt=result['axes']['constraints'][mode]
        assert receipt['gold_label'] is None
        assert receipt['token_cross_entropy'] is None and receipt['eos_cross_entropy'] is None
        assert receipt['target_token_ids'] is None
    assert result['axes']['support']['active']['target_token_ids']==[9,1]


def test_end_to_end_saved_receipt_validation_and_dual_precision(tmp_path):
    from types import SimpleNamespace
    import io
    mod=runner();probe,helper=mod.dependencies()
    verifier=mod.load('collapse_test_verifier',Path(mod.__file__).with_name('verify_generative_quality_reload.py'))
    previous_threads=torch.get_num_threads()
    try:
        verifier.configure_runtime('cpu')
        model=model_fixture()
        data=tmp_path/'data';data.mkdir()
        rows=rows_fixture()
        for row in rows:
            row['question_id']=row['id']
            row['evidence'][0]['source_id']='source'+row['id']
            row['evidence'][0]['text']='hello world '+row['id']
        splits={'train':[],'calibration':rows,'development':[]}
        for split,items in splits.items():
            (data/f'{split}.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in items))
        (data/'manifest.json').write_text(json.dumps({'files':{path.name:verifier.sha256(path) for path in data.glob('*.jsonl')}}))
        run=tmp_path/'run';run.mkdir();model.save_pretrained(run/'model')
        ids={'yes':[8],'no':[9]}
        report={'threshold':.5,'instructions':probe.INSTRUCTIONS,'label_ids':ids,'foundation':model.configuration().get('foundation'),
                'dtype':'float32','autocast_dtype':'bfloat16','max_tokens':256,
                'data_manifest_sha256':verifier.sha256(data/'manifest.json'),
                **{key:model.config[key] for key in ('memory_update','memory_mode','workspace')},
                'training':{'foundation_sha256_after':helper.tensor_digest(model.foundation)},
                'splits':probe.evaluate_splits(model,splits,helper,ids,io.StringIO(),compare_workspace=True,autocast_dtype=torch.bfloat16)}
        (run/'report.json').write_text(json.dumps(report))
        args=SimpleNamespace(run=run,data=data,output=tmp_path/'diagnostic.json',device='cpu')
        result=mod.run(args)
        assert result['selected_saved_receipts_exact']
        assert set(result['records'])=={'float32','bfloat16_autocast'}
        assert all(len(records)==4 for records in result['records'].values())
        assert result['selected_groups']=={'all_known_good':['0','1'],'any_known_false':['3','4']}
        with pytest.raises(FileExistsError):mod.run(args)
        for field,value in [('memory_update','unbounded'),('memory_mode','slots'),('workspace',{'slots':99,'steps':2}),('workspace',{'slots':3,'steps':2.0})]:
            original=report[field];report[field]=value
            (run/'report.json').write_text(json.dumps(report));args.output=tmp_path/f'bad-{field}.json'
            with pytest.raises(ValueError,match='conditioning'):mod.run(args)
            assert not args.output.exists()
            report[field]=original
        report['splits']['calibration']['records'][0]['scores']['support']=.123456789
        (run/'report.json').write_text(json.dumps(report));args.output=tmp_path/'bad.json'
        with pytest.raises(AssertionError,match='receipt'):mod.run(args)
        assert not args.output.exists()
    finally:torch.set_num_threads(previous_threads)


def test_selection_does_not_substitute_unresolved_rows_for_failures():
    rows=rows_fixture()
    for row in rows[3:]:row['targets']['support']=None
    with pytest.raises(ValueError,match='two eligible'):runner().select_rows(model_fixture(),rows)
