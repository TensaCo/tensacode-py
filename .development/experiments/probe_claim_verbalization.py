"""Paired answer-fragment/declarative-claim diagnostic; transformations untrusted."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re


def verbalization_prompt(row):
    if any(not isinstance(row.get(k),str) or not row[k].strip() for k in ('question','candidate')):
        raise ValueError('question and candidate must be nonempty text')
    return ('Combine the question and proposed answer into one self-contained declarative statement. '
            'Keep the proposed answer and every restriction from the question. Do not answer again or add facts. Return only the statement.\n'
            +json.dumps({'question':row['question'],'proposed_answer':row['candidate']},ensure_ascii=False))


def answer_preserved(answer,claim):
    normalized=lambda s:' '.join(s.casefold().split())
    return bool(re.search(r'(?<!\w)'+re.escape(normalized(answer))+r'(?!\w)',normalized(claim)))


def run(args):
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch
    from tensorcode.tools.chatbot import Chatbot
    from tensorcode._internal.cognition.policy import SelectionPolicy
    if not torch.cuda.is_available():raise RuntimeError('real-model diagnostic requires CUDA host')
    if not Path(args.model).is_dir():raise ValueError('explicit local model required')
    rows=[json.loads(line) for line in Path(args.candidates).read_text().splitlines() if line.strip()]
    if not rows or len({row['id'] for row in rows})!=len(rows):raise ValueError('unique nonempty candidates required')
    for row in rows:verbalization_prompt(row)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(8);torch.manual_seed(20260921)
    bot=Chatbot.from_pretrained(args.model,local_files_only=True,device='cuda').eval().requires_grad_(False)
    if bot.investigator.config['verification_scope']!='joint':raise ValueError('owned joint verifier required')
    generator=bot.investigator.generator;policy=SelectionPolicy(**bot.config['cognition']['policy'])
    report={'role':'development claim-verbalization diagnostic; transformations need fidelity review',
            'model':str(Path(args.model).resolve()),'model_fingerprint':bot.fingerprint,
            'candidates_sha256':hashlib.sha256(Path(args.candidates).read_bytes()).hexdigest(),
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'policy':policy.receipt(),'records':[],
            'limitations':['No supervision or source text enters the question/answer verbalizer.',
                           'Literal answer retention does not establish preserved meaning or all restrictions.',
                           'NLI approval does not establish answer correctness; no production admission.']}
    def check(text,evidence):
        receipt=bot.investigator.verify(text,evidence)
        receipt['accepted_by_policy']=policy.accepts_verification(receipt,[e['source_id'] for e in evidence],scope='joint')
        return receipt
    with (output/'progress.jsonl').open('x') as stream,torch.no_grad():
        for index,row in enumerate(rows):
            prompt=verbalization_prompt(row)
            count=len(generator.tokenizer(prompt,truncation=False)['input_ids'])
            record={'input':row,'prompt':prompt,'input_token_count':count,
                    'input_truncated':count>generator.config['max_input_tokens']}
            if not record['input_truncated']:
                claim=generator.generate_batch([prompt])[0]
                record.update(claim=claim,answer_preserved=answer_preserved(row['candidate'],claim),
                              original_verification=check(row['candidate'],row['evidence']))
                if claim.strip():record['claim_verification']=check(claim,row['evidence'])
            report['records'].append(record)
            stream.write(json.dumps(record,ensure_ascii=False)+'\n');stream.flush()
            if (index+1)%16==0:print(json.dumps({'completed':index+1,'total':len(rows)}),flush=True)
    (output/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'rows':len(rows),'answer_preserved':sum(r.get('answer_preserved',False) for r in report['records']),
                      'original_accepted':sum(r.get('original_verification',{}).get('accepted_by_policy',False) for r in report['records']),
                      'claim_accepted':sum(r.get('claim_verification',{}).get('accepted_by_policy',False) for r in report['records'])}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True);p.add_argument('--candidates',required=True);p.add_argument('--output',required=True)
    run(p.parse_args())
