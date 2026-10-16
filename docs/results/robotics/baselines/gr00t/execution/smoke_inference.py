"""Runtime-only smoke; synthetic image/state, not a robot task benchmark."""
import json, time, os
from pathlib import Path
import numpy as np
import torch
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.data.embodiment_tags import EmbodimentTag

torch.set_num_threads(4)
torch.cuda.set_per_process_memory_fraction(0.18)
warmup=torch.zeros(1,device="cuda")
checkpoint=Path('/tmp/gr00t-demo/checkpoint')
statistics=json.loads((checkpoint/'statistics.json').read_text())['unitree_g1']['state']
obs={
    'video': {'ego_view': np.zeros((1,1,256,256,3),dtype=np.uint8)},
    'state': {k:np.array(v['mean'],dtype=np.float32)[None,None,:] for k,v in statistics.items()},
    'language': {'annotation.human.task_description': [['Pick the apple and place it on the plate.']]},
}
start=time.monotonic()
policy=Gr00tPolicy(embodiment_tag=EmbodimentTag.UNITREE_G1,model_path=str(checkpoint),device='cuda')
load=time.monotonic()-start
start=time.monotonic()
with torch.inference_mode():
    action,info=policy.get_action(obs)
torch.cuda.synchronize()
report={'status':'smoke_passed','task_benchmark':False,'inputs':'synthetic black image and model-statistics mean joint state','model':'nvidia/GR00T-N1.6-G1-PnPAppleToPlate','revision':'897d0313a190f46a2cccaeb34077752a0db4b0de','load_seconds':load,'inference_seconds':time.monotonic()-start,'action_shapes':{k:list(v.shape) for k,v in action.items()},'all_finite':all(np.isfinite(v).all().item() for v in action.values()),'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'torch':torch.__version__}
Path('/tmp/gr00t-demo/smoke-result.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
