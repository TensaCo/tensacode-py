"""Prepare predeclared, disjoint final cases without displaying their contents."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'examples'))
from evaluate_cognition import prepare_hotpot


def all_ids(value):
    if isinstance(value, dict):
        if isinstance(value.get('id'), str):
            yield value['id']
        for item in value.values():
            yield from all_ids(item)
    elif isinstance(value, list):
        for item in value:
            yield from all_ids(item)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    repo=Path(__file__).resolve().parents[2]
    prior=json.loads((repo/'docs/results/cognition-hotpot.json').read_text())['data_manifest']
    path=Path(hf_hub_download(prior['dataset'],prior['file'],repo_type='dataset',revision=prior['revision'],token=False))
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if digest!=prior['sha256']:
        raise ValueError('pinned source content mismatch')
    rows=pq.read_table(path).slice(304,32).to_pylist()
    cases=[prepare_hotpot(row) for row in rows]
    if len(cases)!=32 or len({c['id'] for c in cases})!=32:
        raise ValueError('expected exactly 32 unique final questions')
    seen=set()
    for report in (repo/'docs/results').glob('*.json'):
        seen.update(all_ids(json.loads(report.read_text())))
    overlap=seen & {case['id'] for case in cases}
    if overlap:
        raise ValueError(f'previously recorded question IDs overlap: {sorted(overlap)}')
    args.output.mkdir(parents=True,exist_ok=False)
    case_file=args.output/'cases.jsonl'
    case_file.write_text(''.join(json.dumps(case)+'\n' for case in cases))
    manifest={**{k:prior[k] for k in ('dataset','revision','file','sha256')},
              'offset':304,'count':32,'ids':[case['id'] for case in cases],
              'cases_sha256':hashlib.sha256(case_file.read_bytes()).hexdigest(),
              'previous_report_ids_disjoint':True,'role':'fixed final comparison after configuration freeze',
              'limitations':'oracle support; foundation pretraining exposure unknown; disjoint from recorded report IDs only'}
    (args.output/'data-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'count':32,'cases_sha256':manifest['cases_sha256'],'previous_report_ids_disjoint':True}))

if __name__=='__main__':
    main()
