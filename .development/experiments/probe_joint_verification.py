"""Compare joint/source screening on fixed development proposals; no promotion."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from tensorcode.tools.chatbot import Chatbot
from tensorcode._internal.cognition.policy import SelectionPolicy

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--model',required=True);p.add_argument('--candidates',required=True)
p.add_argument('--output',required=True);p.add_argument('--max-tokens',type=int);a=p.parse_args()
if not torch.cuda.is_available():raise RuntimeError('real-model diagnostic requires CUDA host')
torch.set_num_threads(8)
bot=Chatbot.from_pretrained(a.model,device='cuda').eval()
if a.max_tokens is not None:
 bot.investigator.verifier.max_tokens=a.max_tokens
 bot.investigator.verifier.config['verifier_max_tokens']=a.max_tokens
policy=SelectionPolicy();rows=[]
for line in Path(a.candidates).read_text().splitlines():
 r=json.loads(line)
 evidence=[{'source_id':e['id'],'text':e['text']} for e in r['evidence']]
 single=bot.investigator.verifier.verify(r['candidate'],evidence)
 joint=bot.investigator.verifier.verify(r['candidate'],[{'source_id':'joint','text':'\n\n'.join(e['text'] for e in evidence)}])[0]
 source_accept=policy.accepts([v['distribution'] for v in single]) and not any(v['input_truncated'] for v in single)
 joint_accept=policy.accepts([joint['distribution']]) and not joint['input_truncated'] and not any(v['input_truncated'] or v['distribution']['contradiction']>policy.max_contradiction for v in single)
 rows.append({'id':r['id'],'question_index':r['question_index'],'candidate':r['candidate'],'source_accept':source_accept,'joint_accept':joint_accept,'source_verifications':single,'joint_verification':joint,'ordered_source_ids':[e['source_id'] for e in evidence]})
Path(a.output).write_text(json.dumps({'role':'historical development proposals; no fresh final access','model':a.model,'candidates_sha256':hashlib.sha256(Path(a.candidates).read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'policy':policy.receipt(),'max_tokens':bot.investigator.verifier.max_tokens,'joint_premise':'exact source texts joined by two newlines in supplied order','records':rows},indent=2)+'\n')
print(json.dumps({'count':len(rows),'source_accepted':sum(r['source_accept'] for r in rows),'joint_accepted':sum(r['joint_accept'] for r in rows)}))
