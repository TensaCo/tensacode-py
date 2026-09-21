import copy
import importlib.util
from pathlib import Path

import pytest

AXES=('support','completeness','constraints')


def runner():
    path=Path(__file__).parents[2]/'.development/experiments/assess_quality_gate.py'
    spec=importlib.util.spec_from_file_location('quality_gate',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def fixture():
    rows=[{'id':str(i),'targets':dict.fromkeys(AXES,i<5)} for i in range(7)]
    # Exactly 80% positive retention and 50% negative rejection per axis;
    # rotate accepted negative axes so neither known failure passes jointly.
    scores=[dict.fromkeys(AXES,.5) for _ in range(4)]+[dict.fromkeys(AXES,.49),
            dict(support=.5,completeness=.49,constraints=.5),
            dict(support=.49,completeness=.5,constraints=.49)]
    records=[{'id':row['id'],'input_truncated':False,'scores':score} for row,score in zip(rows,scores)]
    return rows,{'threshold':.5,'data_manifest_sha256':'digest','splits':{'development':{'records':records,'excluded':[]}}}


def assess(rows,report):
    return runner().assess_report(report,rows,manifest_sha256='digest')


def test_exact_boundary_pass_is_only_numerical():
    rows,report=fixture()
    result=assess(rows,report)
    gate=result['evaluations']['reported']
    assert gate['numerical_pass']
    assert gate['axes']['support']['positive_retention']==.8
    assert gate['axes']['support']['negative_rejection']==.5
    assert gate['joint']['known_good_retention']==.8
    assert result['qualification'] is False and result['promotion'] is False


def test_known_false_with_other_unknown_is_known_failure():
    rows,report=fixture()
    rows.append({'id':'unknown','targets':{'support':False,'completeness':None,'constraints':None}})
    report['splits']['development']['records'].append({'id':'unknown','input_truncated':False,'scores':dict.fromkeys(AXES,1.)})
    result=assess(rows,report)['evaluations']['reported']
    assert result['joint']['accepted_known_failure']==1
    assert result['joint']['accepted_unresolved']==0
    assert not result['numerical_pass']


def test_missing_negative_examples_cannot_pass():
    rows,report=fixture()
    for row in rows:row['targets']['support']=True
    result=assess(rows,report)['evaluations']['reported']
    assert result['axes']['support']['negative_rejection'] is None
    assert not result['axes']['support']['pass'] and not result['numerical_pass']


def test_exclusion_stays_in_positive_retention_denominator():
    rows,report=fixture()
    block=report['splits']['development']
    block['records'][0].update(input_truncated=True,scores=None)
    block['excluded']=['0']
    result=assess(rows,report)['evaluations']['reported']
    assert result['coverage']=={'total':7,'scored':6,'excluded':1,'fraction':6/7}
    assert result['axes']['support']['positive_retention']==.6
    assert not result['numerical_pass']


@pytest.mark.parametrize('change',[
    lambda r:r.update(data_manifest_sha256='wrong'),
    lambda r:r.update(threshold=.6),
    lambda r:r['splits']['development']['records'][0].update(id='absent'),
    lambda r:r['splits']['development']['records'][0].update(id='1'),
    lambda r:r['splits']['development']['records'][0]['scores'].update(support=float('nan')),
    lambda r:r['splits']['development']['records'][0]['scores'].update(support=float('inf')),
    lambda r:r['splits']['development']['records'][0]['scores'].update(support=True),
    lambda r:r['splits']['development']['records'][0]['scores'].update(extra=.5),
    lambda r:r['splits']['development'].update(excluded=['0']),
])
def test_inconsistent_provenance_or_receipts_rejected(change):
    rows,report=fixture();change(report)
    with pytest.raises(ValueError):assess(rows,report)


def test_before_and_bypass_apply_identical_gate():
    rows,report=fixture()
    for record in report['splits']['development']['records']:
        record['bypass_scores']=dict(record['scores'])
    report['before_training']=copy.deepcopy(report['splits'])
    result=assess(rows,report)
    assert set(result['evaluations'])=={'reported','same_foundation_bypass','before_reported','before_same_foundation_bypass'}
    assert all(gate['numerical_pass'] for gate in result['evaluations'].values())


def test_cli_helper_checks_prepared_file_hashes_and_preserves_output(tmp_path):
    import hashlib
    import json
    from types import SimpleNamespace
    rows,report=fixture()
    for row in rows:
        row.update(question_id='q'+row['id'],question='Question '+row['id'],candidate='answer',
                   evidence=[{'id':'source'+row['id'],'source_id':'source'+row['id'],'text':'Evidence '+row['id']}])
    data=tmp_path/'data';data.mkdir()
    for split,records in [('train',[]),('calibration',[]),('development',rows)]:
        (data/f'{split}.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in records))
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    manifest={'files':{path.name:sha(path) for path in data.glob('*.jsonl')}}
    (data/'manifest.json').write_text(json.dumps(manifest))
    report['data_manifest_sha256']=sha(data/'manifest.json')
    report_path=tmp_path/'report.json';report_path.write_text(json.dumps(report))
    output=tmp_path/'gate.json'
    args=SimpleNamespace(report=report_path,data=data,output=output,split='development')
    result=runner().run(args)
    assert result['evaluations']['reported']['numerical_pass']
    assert result['report_sha256']==sha(report_path)
    with pytest.raises(FileExistsError):runner().run(args)
    args.output=tmp_path/'second.json'
    with (data/'development.jsonl').open('a') as stream:stream.write('\n')
    with pytest.raises(ValueError,match='checksum'):runner().run(args)
    assert not args.output.exists()


def test_no_known_good_pool_and_unresolved_acceptance_are_not_certified():
    rows,report=fixture()
    for row in rows:row['targets']['constraints']=None
    result=assess(rows,report)['evaluations']['reported']
    assert result['joint']['known_good_total']==0
    assert result['joint']['known_good_retention'] is None
    assert result['joint']['accepted_unresolved']==4
    assert not result['joint']['pass'] and not result['numerical_pass']


def test_partial_bypass_receipts_are_rejected():
    rows,report=fixture()
    report['splits']['development']['records'][0]['bypass_scores']=dict.fromkeys(AXES,.5)
    with pytest.raises(ValueError,match='bypass'):assess(rows,report)
