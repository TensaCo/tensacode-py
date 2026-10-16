"""Reproduce bounded G1 substrate fidelity and physical grasp-control evaluations."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import onnxruntime as ort
import torch
from humanoid_policy import compile_walker, main as rollout


def main():
 p=argparse.ArgumentParser();p.add_argument('--assets',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 torch.set_num_threads(2)
 artifact=compile_walker(a.assets)
 (a.output/'walker.json').write_text(artifact.freeze().to_json())
 options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
 policy_path=a.assets/'deploy/robots/g1/config/policy/velocity/v0/exported/policy.onnx'
 native=ort.InferenceSession(str(policy_path),sess_options=options,providers=['CPUExecutionProvider'])
 records=json.loads((a.reference/'records.json').read_text());cases=[np.asarray(r['observation'],np.float32) for r in records[::max(1,len(records)//64)]]
 fidelity={}
 for target in ['alg0','alg1','alg2']:
  machine=artifact.compile(target);max_error=0.
  with torch.no_grad():
   for x in cases:
    expected=native.run(None,{'obs':x[None]})[0][0];actual=machine({'observation':x})['action'];actual=actual.numpy() if isinstance(actual,torch.Tensor) else actual
    max_error=max(max_error,float(np.max(np.abs(expected-actual))))
    if not np.allclose(expected,actual,atol=2e-5,rtol=2e-5):raise RuntimeError(f'Fidelity failed for {target}: {max_error}')
  fidelity[target]={'cases':len(cases),'max_abs_error':max_error,'atol':2e-5,'rtol':2e-5,'passed':True}
  print(json.dumps({'fidelity':target,**fidelity[target]}),flush=True)
 del machine
 results={'native':json.loads((a.reference/'probe.json').read_text())}
 tasks=[('alg0','alg0',False,1.25,-.25,.08),('alg1','alg1',False,1.25,-.25,.08),('alg2','alg2',False,1.25,-.25,.08),('open-gripper','native',True,1.25,-.25,.08),('shifted-light','alg0',False,1.28,-.23,.06),('shifted-heavy','alg0',False,1.22,-.27,.10)]
 for name,target,open_gripper,x,y,mass in tasks:
  rollout(SimpleNamespace(assets=a.assets,output=a.output/name,task=True,substrate=target,seconds=30.,speed=.3,no_gripper=False,open_gripper=open_gripper,object_x=x,object_y=y,mass=mass))
  results[name]=json.loads((a.output/name/'probe.json').read_text())
  (a.output/'interim.json').write_text(json.dumps({'fidelity':fidelity,'rollouts':results},indent=2))
 report={'fidelity':fidelity,'rollouts':results,'provenance':{'source_repository':'https://github.com/unitreerobotics/unitree_rl_mjlab','source_revision':'1425b15f73bd4095f0df53709d7c389c3eb9e790','policy_sha256':hashlib.sha256(policy_path.read_bytes()).hexdigest(),'walker_sha256':hashlib.sha256((a.output/'walker.json').read_bytes()).hexdigest(),'versions':{n:importlib.metadata.version(n) for n in ['torch','numpy','mujoco','onnxruntime','onnx','scipy']},'example_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('humanoid_policy.py')]}},'training':'No new training: supplied Unitree policy, authored manipulation and servo controller.','physics':{'free_base':True,'object_attachment':False,'object_pose_overwritten_during_simulation':False,'robot_self_collisions':False,'gripper':'authored two-jaw attachment, not stock G1 hand','dt':.002,'policy_dt':.02},'scope':'Single morphology and three scripted placements/masses; ground-truth state and object pose supplied; no perception or hardware claim.'}
 (a.output/'report.json').write_text(json.dumps(report,indent=2))
 if not all(results[n]['success'] for n in ['native','alg0','alg1','alg2']):raise RuntimeError('Required nominal closed-loop run failed')
 if results['open-gripper']['max_lift']>.02:raise RuntimeError('Open-gripper control unexpectedly lifted object')
 print('Required fidelity, closed-loop and open-gripper controls passed.',flush=True)

if __name__=='__main__':main()
