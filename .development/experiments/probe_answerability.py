"""Question-conditioned score diagnostic; no production policy or model changes."""
import argparse
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--report',required=True);p.add_argument('--output',required=True);a=p.parse_args()
model=AutoModelForSequenceClassification.from_pretrained(a.model,local_files_only=True,use_safetensors=True,trust_remote_code=False).to('cuda').eval()
tokenizer=AutoTokenizer.from_pretrained(a.model,local_files_only=True,trust_remote_code=False)
records=json.loads(Path(a.report).read_text())['real_data']['records'];out=[]
for r in records:
 texts=[c['text'] for c in r['receipt']['cognition']['candidates']]+[r['answer']]
 inputs=tokenizer([r['question']]*len(texts),texts,padding=True,truncation=True,max_length=512,return_tensors='pt')
 with torch.no_grad():
  logits=model(**{k:v.to('cuda') for k,v in inputs.items()}).logits
 if logits.shape!=(len(texts),1):raise ValueError('expected documented one-logit binary answerability head')
 scores=logits.sigmoid().flatten().cpu().tolist()
 out.append({'id':r['id'],'question':r['question'],'target':r['target'],
             'candidates':[{'id':c['id'],'text':c['text'],'score':s,'source_policy_accept':c['accepted_by_policy']} for c,s in zip(r['receipt']['cognition']['candidates'],scores)],
             'answer':r['answer'],'abstention_enforced':r['abstention_enforced'],'answer_score':scores[-1],
             'input_truncated':[len(tokenizer(r['question'],text)['input_ids'])>512 for text in texts]})
Path(a.output).write_text(json.dumps({'role':'development only','foundation_revision':'c7dea87c98b2269a935686c31336e97e837cbbeb','threshold':0.5,'records':out},indent=2))
