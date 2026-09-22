"""Collect a text -> OUTPUT_ENCODING -> text program, train, and reload its ops.

Supply reviewed JSONL rows with text and target fields. Run real foundations on
an appropriate training host. The loss is measured on supplied training examples,
not held-out generalization; native foundation weights remain frozen. This example
saves complete operation weights, not optimizer/RNG continuation.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def run(args):
    import torch
    from tensorcode import trace, training
    from tensorcode.ops.vec import latent_codecs
    from tensorcode.ops.vec.encode import TextEncoder
    from tensorcode.ops.vec.decode import TextDecoder
    if args.steps<1 or args.batch_size<1 or not 0<args.lr<1:
        raise ValueError('positive steps/batch_size and lr in (0,1) required')
    data=Path(args.data)
    rows=[json.loads(line) for line in data.read_text().splitlines() if line.strip()]
    if not rows or any(not isinstance(r,dict) or set(r)!={'text','target'} or
                       any(not isinstance(r[k],str) or not r[k].strip() for k in r) for r in rows):
        raise ValueError('supply nonempty JSONL rows containing text and target strings')
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(17)
    # All trainable parameters exist before capture. Import local foundations explicitly.
    encode=TextEncoder.from_foundation(args.foundation,revision=args.revision,
        local_files_only=True,readout='output_encoding').to(args.device).eval()
    decode=TextDecoder.from_foundation(args.foundation,revision=args.revision,
        local_files_only=True,input_space=encode.output_space,bridge='linear',
        generation={'max_new_tokens':32}).to(args.device).eval()
    encode.model.requires_grad_(False);decode.model.requires_grad_(False)
    if any(len(encode.tokenizer(r['text'])['input_ids'])>255 or
           len(decode.tokenizer(r['target'])['input_ids'])>128 for r in rows):
        raise ValueError('example budget is 255 input and 128 target tokens; split long examples explicitly')
    operations={'encode':encode,'objective':decode.training_operation}
    trainer=training.Trainer.from_ops(operations,optimizer=lambda ps:torch.optim.AdamW(ps,lr=args.lr),
                             losses={'objective':lambda loss,target:loss})
    before_readout=encode.output_encoding.detach().clone()
    before_bridge=decode.projection.weight.detach().clone()
    source='supplied JSONL supervision sha256:'+hashlib.sha256(data.read_bytes()).hexdigest()
    paths=[]
    for start in range(0,len(rows),args.batch_size):
        batch=rows[start:start+args.batch_size]
        with trace() as experience:
            latent=encode([r['text'] for r in batch])
            loss=decode.training_operation({'inputs':latent,'targets':[r['target'] for r in batch]})
        experience.supervise(loss,[r['target'] for r in batch],loss='objective',source=source)
        path=output/f'experience-{len(paths):05d}.json'
        experience.save(path,operations=operations,codecs=latent_codecs(),release=True)
        paths.append(path)
    losses=[]
    for step in range(args.steps):
        experience=training.load_experience(paths[step%len(paths)],operations=operations,codecs=latent_codecs())
        losses.append(trainer.step(experience))
    encode.save_pretrained(output/'encoder');decode.save_pretrained(output/'decoder')
    sample=rows[:args.batch_size]
    with torch.no_grad():
        expected=decode.loss(encode([r['text'] for r in sample]),[r['target'] for r in sample])
    restored_encode=TextEncoder.from_pretrained(output/'encoder',device=args.device)
    restored_decode=TextDecoder.from_pretrained(output/'decoder',device=args.device)
    with torch.no_grad():
        actual=restored_decode.loss(restored_encode([r['text'] for r in sample]),[r['target'] for r in sample])
    report={'examples':len(rows),'steps':args.steps,'losses':losses,
        'readout_changed':not torch.equal(before_readout,encode.output_encoding),
        'bridge_changed':not torch.equal(before_bridge,decode.projection.weight),
        'reloaded_loss_exact':torch.equal(expected,actual),'supervision_sha256':source.split(':')[-1],
        'foundation':args.foundation,'revision':args.revision,
        'limitations':'Supplied training pairs only; frozen native foundations; no shared-semantic-space or generalization claim; weights-only reload.'}
    if not report['reloaded_loss_exact']:raise AssertionError('restored operation loss differs')
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--foundation',required=True,help='local safetensors foundation directory')
    p.add_argument('--revision',required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda')
    p.add_argument('--steps',type=int,default=50);p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--lr',type=float,default=.001)
    print(json.dumps(run(p.parse_args()),indent=2))
