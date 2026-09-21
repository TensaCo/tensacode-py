import importlib.util,json,os
from pathlib import Path
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['HF_HUB_OFFLINE']='1'
import torch
from tensorcode.tools.chatbot import Chatbot
root=Path('/home/brandonin/tensorcode-runs/cognition-20260921')
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
probe=load('probe',root/'repo/.development/experiments/probe_generative_quality.py')
helper=load('helper',root/'repo/examples/train_response_quality.py')
run=root/'artifacts/quality-v2-data/workspace-xl'
report=json.loads((run/'report.json').read_text())
torch.set_num_threads(8);torch.use_deterministic_algorithms(True)
torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
model=Chatbot.from_pretrained(run/'model',device='cuda').eval()
_,splits=helper.load_data(root/'artifacts/quality-v2-data/prepared')
count=0
for split in ('calibration','development'):
 for row,expected in zip(splits[split],report['splits'][split]['records'],strict=True):
  assert row['id']==expected['id']
  result=probe.assess(model,helper.model_inputs(row),yes_id=report['label_ids']['yes'][0],no_id=report['label_ids']['no'][0],workspace_ablation=None)
  for key,value in result.items():assert value==expected[key],(row['id'],key,value,expected[key])
  if not result['input_truncated']:
   bypass=probe.assess(model,helper.model_inputs(row),yes_id=report['label_ids']['yes'][0],no_id=report['label_ids']['no'][0])
   assert bypass['scores']==expected['bypass_scores'],row['id']
  count+=1
 print(split,count,flush=True)
result={'records':count,'active_and_bypass_exact':True,'raw_report_sha256':helper.sha256(run/'report.json'),'verification_script_sha256':helper.sha256(__file__)}
(run/'fresh-reload-verification.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
