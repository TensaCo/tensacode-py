"""Read-only verifier interventions on historical receipts (development data)."""
import argparse
import json
import re
from pathlib import Path
import torch
from tensorcode.tools.chatbot import Chatbot
from tensorcode._internal.cognition.policy import SelectionPolicy

p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--report',required=True);p.add_argument('--output',required=True);a=p.parse_args()
torch.manual_seed(17)
bot=Chatbot.from_pretrained(a.model,device='cuda'); verifier=bot.investigator.verifier
policy=SelectionPolicy(**bot.config['cognition']['policy'])
report=json.loads(Path(a.report).read_text()); records=[]
for case in report['real_data']['records']:
 cognition=case['receipt']['cognition']; sources=cognition['evidence']
 evidence=[{'source_id': x['id'], 'text':x['text']} for x in sources]
 for candidate in cognition['candidates']:
  baseline=verifier.verify(candidate['text'],evidence)
  baseline_accept=policy.accepts([x['distribution'] for x in baseline]) and not any(x['input_truncated'] for x in baseline)
  chunks=[]
  for source in evidence:
   # Authored punctuation segmentation diagnostic, never claimed semantic parsing.
   parts=[x for x in re.split(r'(?<=[.!?])\s+|\n+',source['text']) if x.strip()]
   chunks.extend({'source_id':source['source_id']+':'+str(i),'text':text} for i,text in enumerate(parts))
  window=verifier.verify(candidate['text'],chunks)
  window_accept=policy.accepts([x['distribution'] for x in window]) and not any(x['input_truncated'] for x in window)
  item={'id':case['id'],'question':case['question'],'target':case['target'],'candidate':candidate['text'],'candidate_id':candidate['id'],
        'baseline_accept':baseline_accept,'recorded_accept':candidate['accepted_by_policy'],'window_accept':window_accept,
        'baseline':baseline,'window':window,'chunks':chunks,'sources':evidence}
  records.append(item)
 print(json.dumps({'id':case['id'],'baseline':sum(x['baseline_accept'] for x in records if x['id']==case['id']), 'window':sum(x['window_accept'] for x in records if x['id']==case['id'])}),flush=True)
 Path(a.output).write_text(json.dumps({'role':'historical evaluation reused for development only','intervention':'same model, temperature and thresholds; punctuation-delimited source segments; max contradiction across all segments', 'records':records},indent=2))
