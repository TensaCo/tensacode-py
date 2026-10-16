import os,time,json,random
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl'); os.environ.setdefault('PYOPENGL_PLATFORM','egl'); os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE','1')
import numpy as np, torch, psutil
from gr00t.policy.gr00t_policy import Gr00tPolicy,Gr00tSimPolicyWrapper
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.eval.rollout_policy import run_rollout_gymnasium_policy,WrapperConfigs,VideoConfig,MultiStepConfig
random.seed(11); np.random.seed(11); torch.manual_seed(11); torch.set_num_threads(4)
torch.cuda.set_per_process_memory_fraction(.18)
warm=torch.zeros(1,device='cuda')
out=Path('/tmp/gr00t-demo/episode');out.mkdir(exist_ok=True)
started=time.monotonic()
policy=Gr00tSimPolicyWrapper(Gr00tPolicy(embodiment_tag=EmbodimentTag.UNITREE_G1,model_path='/tmp/gr00t-demo/checkpoint',device='cuda'))
trace=[]; original=policy.get_action
def get_action(obs,*args,**kwargs):
 mem=psutil.virtual_memory()
 if mem.available/mem.total<.2:raise RuntimeError('20% available memory guard reached')
 t=time.monotonic()
 result=original(obs,*args,**kwargs)
 torch.cuda.synchronize()
 trace.append({'call':len(trace),'seconds':time.monotonic()-t,'available_ram_bytes':mem.available,'total_ram_bytes':mem.total,'cuda_allocated_bytes':torch.cuda.memory_allocated(),'cuda_peak_bytes':torch.cuda.max_memory_allocated()})
 (out/'progress.json').write_text(json.dumps(trace,indent=2))
 print('POLICY_CALL',trace[-1],flush=True)
 return result
policy.get_action=get_action
cfg=WrapperConfigs(video=VideoConfig(video_dir=str(out/'video'),max_episode_steps=1440,steps_per_render=2,fps=20,thread_count=1),multistep=MultiStepConfig(n_action_steps=20,max_episode_steps=1440,terminate_on_success=True))
result=run_rollout_gymnasium_policy(env_name='gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc',policy=policy,wrapper_configs=cfg,n_episodes=1,n_envs=1)
report={'model':'nvidia/GR00T-N1.6-G1-PnPAppleToPlate','revision':'897d0313a190f46a2cccaeb34077752a0db4b0de','environment':result[0],'successes':[bool(x) for x in result[1]],'episode_info':result[2],'elapsed_seconds':time.monotonic()-started,'policy_calls':len(trace),'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'runtime_changes':['PyTorch2.14CUDA13 read-only reused from existingvenv','SDPA attention replaces flash-attn due wheelABIincompatibility','checkpoint loaded directlyBF16','Python3.12dataclassdefault_factory compatibility fix'],'comparison':'Separate official apple-to-plate task, not a matched comparison with TensorCode warehouse scene.'}
(out/'result.json').write_text(json.dumps(report,indent=2,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x)))
print('EPISODE_RESULT',json.dumps(report,default=str),flush=True)
