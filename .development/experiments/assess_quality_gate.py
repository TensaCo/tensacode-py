"""Check the fixed development numerical gate; never qualify or promote a model."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

AXES=('support','completeness','constraints')


def unique_ids(values, description):
    if not isinstance(values,list) or any(not isinstance(value,str) or not value for value in values):
        raise ValueError(f'{description} must be nonempty string IDs')
    if len(set(values))!=len(values):
        raise ValueError(f'{description} must have unique IDs')
    return set(values)


def scores_valid(scores):
    return (isinstance(scores,dict) and set(scores)==set(AXES)
            and all(type(value) in (int,float) and math.isfinite(value) and 0<=value<=1 for value in scores.values()))


def validate_block(block, rows):
    if not isinstance(block,dict) or not isinstance(block.get('records'),list):
        raise ValueError('split requires records')
    records=block['records']
    if any(not isinstance(record,dict) for record in records):raise ValueError('records must be objects')
    wanted=unique_ids([row.get('id') for row in rows],'data')
    actual=unique_ids([record.get('id') for record in records],'report records')
    if actual!=wanted:raise ValueError('report records must cover exact data IDs')
    excluded=unique_ids(block.get('excluded'),'excluded')
    truncated=set()
    for record in records:
        if type(record.get('input_truncated')) is not bool:raise ValueError('input_truncated must be boolean')
        if record['input_truncated']:
            truncated.add(record['id'])
            if record.get('scores') is not None or record.get('bypass_scores') is not None:
                raise ValueError('truncated rows must have no scores')
        elif not scores_valid(record.get('scores')):
            raise ValueError('scores require exactly three finite probabilities')
    if excluded!=truncated:raise ValueError('excluded IDs must exactly match truncated records')
    has_bypass=any('bypass_scores' in record for record in records)
    if has_bypass and any(not scores_valid(record.get('bypass_scores')) for record in records if not record['input_truncated']):
        raise ValueError('bypass scores must cover every scored record')
    return {record['id']:record for record in records},has_bypass


def gate(rows,records,*,score_key='scores'):
    # Missing model output counts as rejection, with every labelled row retained
    # in the denominator. Unknown labels never certify either truth or failure.
    decisions={row['id']:{axis:False if records[row['id']]['input_truncated']
                         else records[row['id']][score_key][axis]>=.5 for axis in AXES} for row in rows}
    axes={}
    for axis in AXES:
        positives=[row for row in rows if row['targets'][axis] is True]
        negatives=[row for row in rows if row['targets'][axis] is False]
        retained=sum(decisions[row['id']][axis] for row in positives)
        rejected=sum(not decisions[row['id']][axis] for row in negatives)
        axes[axis]={'known_positive':len(positives),'known_negative':len(negatives),
                    'unknown':len(rows)-len(positives)-len(negatives),
                    'retained_positive':retained,'rejected_negative':rejected,
                    'positive_retention':retained/len(positives) if positives else None,
                    'negative_rejection':rejected/len(negatives) if negatives else None,
                    'pass':bool(positives and negatives and retained*5>=len(positives)*4 and rejected*2>=len(negatives))}
    accepted=[row for row in rows if all(decisions[row['id']].values())]
    good=[row for row in rows if all(value is True for value in row['targets'].values())]
    retained_good=sum(all(value is True for value in row['targets'].values()) for row in accepted)
    failures=sum(any(value is False for value in row['targets'].values()) for row in accepted)
    unresolved=len(accepted)-retained_good-failures
    joint={'known_good_total':len(good),'accepted':len(accepted),'accepted_known_good':retained_good,
           'accepted_known_failure':failures,'accepted_unresolved':unresolved,
           'known_good_retention':retained_good/len(good) if good else None,
           'pass':bool(good and retained_good*2>=len(good) and failures==0)}
    scored=sum(not record['input_truncated'] for record in records.values())
    return {'numerical_pass':all(value['pass'] for value in axes.values()) and joint['pass'],
            'axes':axes,'joint':joint,
            'coverage':{'total':len(rows),'scored':scored,'excluded':len(rows)-scored,
                        'fraction':scored/len(rows) if rows else None}}


def assess_report(report,rows,*,manifest_sha256,split='development'):
    if not isinstance(report,dict):raise ValueError('report must be an object')
    if report.get('data_manifest_sha256')!=manifest_sha256:raise ValueError('report data manifest checksum mismatch')
    if type(report.get('threshold')) not in (int,float) or report['threshold']!=.5:
        raise ValueError('only the fixed .5 threshold protocol is supported')
    if not isinstance(rows,list) or any(not isinstance(row,dict) for row in rows):raise ValueError('rows must be objects')
    for row in rows:
        targets=row.get('targets')
        if not isinstance(targets,dict) or set(targets)!=set(AXES) or any(value is not None and type(value) is not bool for value in targets.values()):
            raise ValueError('targets must specify every axis as bool or None')
    evaluations={}
    sources=[('',report.get('splits'))]
    if 'before_training' in report:sources.append(('before_',report['before_training']))
    for prefix,splits in sources:
        if not isinstance(splits,dict) or split not in splits:raise ValueError('requested split missing from report')
        records,has_bypass=validate_block(splits[split],rows)
        evaluations[prefix+'reported']=gate(rows,records)
        if has_bypass:evaluations[prefix+'same_foundation_bypass']=gate(rows,records,score_key='bypass_scores')
    return {'role':'fixed numerical gate assessment only','split':split,'threshold':.5,
            'data_manifest_sha256':manifest_sha256,'evaluations':evaluations,
            'qualification':False,'promotion':False,
            'remaining_requirements':['Reviewed evidence-sensitivity interventions.',
                                      'Complete public-tool behavior and source-reviewed controls.',
                                      'Freeze the pipeline before reserved final evaluation.'],
            'limitations':['Assistant-reviewed development labels are not human ground truth.',
                           'Unknown labels remain unresolved; numerical passage is provisional.',
                           'Excluded/truncated rows count as rejection in all retention denominators.',
                           'Only the development split is the predeclared integration gate.']}


def run(args):
    helper_path=Path(__file__).resolve().parents[2]/'examples/train_response_quality.py'
    spec=importlib.util.spec_from_file_location('quality_helpers',helper_path)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    _,splits=helper.load_data(args.data)
    report=json.loads(Path(args.report).read_text())
    result=assess_report(report,splits[args.split],manifest_sha256=helper.sha256(Path(args.data)/'manifest.json'),split=args.split)
    result['report_sha256']=helper.sha256(args.report)
    result['script_sha256']=helper.sha256(__file__)
    with Path(args.output).open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({name:value['numerical_pass'] for name,value in result['evaluations'].items()}))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',required=True)
    parser.add_argument('--data',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--split',choices=('development','calibration'),default='development')
    run(parser.parse_args())
