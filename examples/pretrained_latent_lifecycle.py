"""Bounded, explicitly supervised text-latent lifecycle on a pretrained model.

Run on the designated GPU host, for example::

    python examples/pretrained_latent_lifecycle.py --output /tmp/latent-run

The four authored pairs demonstrate adapter optimization, not held-out ability.
Native token embeddings and final encoder states are different Spaces. Only the
former support a native identity decoder; the latter require a trained bridge.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

REVISION = '0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab'


def run(args):
    import torch
    from tensorcode.ops.vec import Space, latent_codecs
    from tensorcode.ops.vec.encode import TextEncoder
    from tensorcode.ops.vec.decode import TextDecoder
    from tensorcode.training import Trainer
    from tensorcode.training import load_experience
    torch.manual_seed(17)
    torch.set_num_threads(8)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    native_space = Space(f'{args.foundation}:encoder:input_embeddings', 512,
                         version=args.revision, organization='sequence')
    identity = TextDecoder.from_foundation(args.foundation, revision=args.revision,
        input_space=native_space, bridge='identity', generation={'max_new_tokens':24},
        local_files_only=args.local_files_only).to(args.device).eval()
    prompts = ['Translate to German: The house is wonderful.',
               'What is the capital of France?', 'What is 2 plus 2?',
               'Answer briefly: What color is a ripe banana?']
    targets = ['Das Haus ist wunderbar.', 'Paris', '4', 'yellow']
    with torch.no_grad():
        embedded = identity.embed_text(prompts)
        wrapped = identity(embedded)
        tokens = identity.tokenizer(prompts, padding=True, return_tensors='pt')
        ids = identity.model.generate(**{k:v.to(args.device) for k,v in tokens.items()
            if k in ('input_ids','attention_mask')}, **identity.generation)
        native = identity.tokenizer.batch_decode(ids, skip_special_tokens=True)
    encoder = TextEncoder.from_foundation(args.foundation, revision=args.revision,
        local_files_only=args.local_files_only).to(args.device).eval()
    with torch.no_grad():
        latent = encoder(prompts)
        hidden = encoder.model.get_encoder()(**{k:v.to(args.device) for k,v in tokens.items()
            if k in ('input_ids','attention_mask')}).last_hidden_state
    rejected = False
    try:
        identity(latent)
    except ValueError:
        rejected = True
    decoder = TextDecoder.from_foundation(args.foundation, revision=args.revision,
        input_space=latent.space, bridge='linear', generation={'max_new_tokens':24},
        local_files_only=args.local_files_only).to(args.device)
    decoder.model.requires_grad_(False)
    optimizer = torch.optim.AdamW(decoder.projection.parameters(), lr=0.001)
    trainer = Trainer.from_tool(decoder, optimizer=optimizer)
    # Disable foundation dropout for this fixed-data adapter diagnostic.
    decoder.model.eval()
    before = float(decoder.loss(latent, targets).detach())
    experience = trainer.capture(latent, targets, source='four authored demonstration pairs; no held-out split')
    experience.save(output/'experience.json', operations=trainer.operations, codecs=latent_codecs())
    replay = load_experience(output/'experience.json', operations=trainer.operations, codecs=latent_codecs())
    losses = [float(trainer.step(replay)) for _ in range(args.steps)]
    decoder.eval()
    after = float(decoder.loss(latent, targets).detach())
    prediction = decoder(latent)
    # Evaluator-only native token inspection records whether generation hit its cap.
    with torch.no_grad():
        adapter_ids = decoder.model.generate(inputs_embeds=decoder.projection(latent.tensor),
            attention_mask=latent.mask, **decoder.generation)
    decoder.save_pretrained(output/'decoder')
    trainer.save_checkpoint(output/'training', progress={'steps':args.steps,'data':'authored fixed pairs'})
    restored = TextDecoder.from_pretrained(output/'decoder', device=args.device)
    restored.model.requires_grad_(False)
    resumed = Trainer.from_tool(restored, optimizer=torch.optim.AdamW(restored.projection.parameters(), lr=0.001))
    progress = resumed.load_checkpoint(output/'training')
    restored.eval()
    def generated_lengths(rows, eos):
        return [int((row == eos).nonzero()[0]) if bool((row == eos).any()) else len(row)-1 for row in rows]
    report = {'foundation':args.foundation,'revision':args.revision,'device':args.device,
        'prompts':prompts,'authored_targets':targets,'native_predictions':native,
        'identity_predictions':wrapped,'identity_native_equal':wrapped==native,
        'encoder_native_max_abs_error':float((latent.tensor-hidden).abs().max()),
        'encoder_final_hidden_identity_rejected':rejected,
        'input_tokens':tokens['attention_mask'].sum(1).tolist(),
        'truncation':{'input':False,'target':False,'generation_max_new_tokens':24,
            'native_generation_missing_eos':[not bool((row == identity.model.config.eos_token_id).any()) for row in ids],
            'adapter_generation_missing_eos':[not bool((row == decoder.model.config.eos_token_id).any()) for row in adapter_ids],
            'native_generated_tokens':generated_lengths(ids, identity.model.config.eos_token_id),
            'adapter_generated_tokens':generated_lengths(adapter_ids, decoder.model.config.eos_token_id)},
        'adapter':{'steps':args.steps,'before_cross_entropy':before,'step_losses':losses,
                   'after_cross_entropy':after,'predictions':prediction,
                   'reload_equal':prediction==restored(latent), 'resumed_steps':resumed.steps,
                   'resumed_progress':progress, 'trainable_parameters':sum(p.numel() for p in decoder.projection.parameters())},
        'limitations':['Native behavior is inherited from supplied FLAN-T5 weights.',
            'Only a linear bridge learned from four authored pairs; no held-out generalization claim.',
            'Final encoder states are not native decoder input embeddings.']}
    (output/'results.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    return report


def image_example(args):
    """Optional pretrained image operations with explicit native conditioning.

    CLIP is used only to supply the foundation's documented conditioning space;
    arbitrary TextEncoder output has not been aligned to this image decoder.
    """
    import torch
    from PIL import Image
    from transformers import CLIPTextModel, CLIPTokenizer
    from tensorcode.ops.vec import Latent, Space
    from tensorcode.ops.vec.encode import ImageEncoder
    from tensorcode.ops.vec.decode import ImageDecoder
    output = Path(args.output) / 'images'
    output.mkdir()
    vit_revision = 'b4569560a39a0f1af58e3ddaf17facf20ab919b0'
    sd_revision = 'b261bac6fd2cf515557d5d0707481eafa0485ec2'
    encoder = ImageEncoder.from_foundation(args.vision_foundation,
        revision=vit_revision, local_files_only=args.local_files_only,
        output_space=Space('vit-patch-states', 768, version=vit_revision, organization='sequence'),
        device=args.device)
    with torch.no_grad():
        features = encoder(encoder.preprocess(Image.open(args.image_input).convert('RGB')))
    encoder.save_pretrained(output/'encoder')
    # This explicit native Space assertion is backed by the same pinned CLIP
    # component and tokenizer. The decoder itself owns its UNet and VAE weights.
    space = Space('sd-turbo-native-clip', 1024, version=sd_revision, organization='sequence')
    decoder = ImageDecoder.from_foundation(args.image_foundation,
        revision=sd_revision, local_files_only=args.local_files_only,
        input_space=space, bridge='identity', num_inference_steps=4).to(args.device)
    options = dict(revision=sd_revision, local_files_only=args.local_files_only)
    text = CLIPTextModel.from_pretrained(args.image_foundation,
        subfolder='text_encoder', use_safetensors=True, **options).to(args.device).eval()
    tokenizer = CLIPTokenizer.from_pretrained(args.image_foundation, subfolder='tokenizer', **options)
    prompt = 'A photograph of a red ceramic teapot on a wooden table in a sunlit kitchen.'
    tokens = tokenizer([prompt], padding='max_length', max_length=tokenizer.model_max_length,
                       truncation=True, return_tensors='pt')
    with torch.no_grad():
        states = text(tokens.input_ids.to(args.device), attention_mask=(
            tokens.attention_mask.to(args.device) if getattr(text.config, 'use_attention_mask', False) else None))[0]
    # Native SD attends to all sequence positions, including EOS padding states.
    conditioning = Latent(states, space, metadata={'representation':'native CLIP conditioning'})
    pixels = decoder(conditioning, context={'seed':17})
    Image.fromarray((pixels[0].permute(1,2,0).cpu().numpy()*255).round().astype('uint8')).save(output/'generated.png')
    decoder.save_pretrained(output/'decoder')
    restored = ImageDecoder.from_pretrained(output/'decoder', device=args.device)
    again = restored(conditioning, context={'seed':17})
    result = {'vision_feature_shape':list(features.tensor.shape), 'prompt':prompt,
              'diffusion_steps':4, 'scheduler':'DDIM', 'seed':17,
              'reload_max_abs_error':float((pixels-again).abs().max()),
              'limitations':['Pretrained representations and synthesis are inherited.',
                  'No image training or arbitrary cross-space alignment is demonstrated.',
                  'DDIM replaces the foundation default scheduler.']}
    (output/'results.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--foundation', default='google/flan-t5-small')
    parser.add_argument('--revision', default=REVISION)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--steps', type=int, default=4)
    parser.add_argument('--local-files-only', action='store_true')
    parser.add_argument('--image-input', help='Optional image path enabling the pretrained image example')
    parser.add_argument('--vision-foundation', default='google/vit-base-patch16-224-in21k')
    parser.add_argument('--image-foundation', default='stabilityai/sd-turbo')
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive')
    run(args)
    if args.image_input:
        image_example(args)
