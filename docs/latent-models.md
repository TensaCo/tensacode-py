# Pretrained vector encoders and decoders

The 0.4 alpha adds owned transformer encoders and text/image generation from
space-tagged vectors. A configuration constructor initializes parameters without
network access. `from_foundation(...)` explicitly imports native model weights;
`from_pretrained(...)` restores a complete TensorCode operation, including adapters.

```bash
python -m pip install -e '.[pretrained]' # text transformers and ViT
python -m pip install -e '.[diffusion]'  # also latent diffusion decoding
```

Heavy examples should run on a suitable training device. The examples below use
local foundation directories so downloads and device placement stay explicit.
Remote foundation IDs also work; pin `revision` for reproducible acquisitions.
Only safetensors foundation weights are accepted.

## Encode with a real transformer

```python
from tensorcode.ops import vec

encode = vec.TextEncoder.from_foundation(
    './flan-t5-small', local_files_only=True, pooling='sequence',
)
latent = encode('The service recovered after reconnecting the database.',
                context={'texts': ['Summarize the incident.']})
print(latent.tensor.shape, latent.space, latent.mask)
encode.save_pretrained('./text-encoder')
restored = vec.TextEncoder.from_pretrained('./text-encoder')
```

Text encoding returns final native transformer states, with a batch axis and a
boolean token mask. `pooling='mean'` instead returns a masked mean in a feature
space. Text context is an ordered list of shared prefixes, joined before tokenizing.
It is part of the actual transformer input. Neither readout is a newly learned
universal cognitive space. A custom `OUTPUT_ENCODING` readout token is not
implemented; the alpha preserves existing native readout behavior.

`vec.ImageEncoder` currently supports **ViT**. It owns the model and processor.
Supply a sequence `Space` matching the ViT hidden width, or use `output='pooled'`
with a feature space for the native final CLS state:

```python
vision = vec.ImageEncoder.from_foundation(
    './vit-base', space=vec.Space('my-vit/patches', 768, organization='sequence'),
    local_files_only=True,
)
processed = vision.preprocess(images)  # supplied PIL images or processor inputs
patches = vision(processed)
```

Raw CHW/BCHW floating tensors in `[0,1]` must already match the configured image
size; their normalization remains differentiable. `preprocess` performs the saved
processor's resize/normalization and makes no input-gradient promise. Patch
coordinates refer to the **processed image**, not invented original-image regions.
Optional `context_space` and `context={'latents': [...]}` append compatible vectors
before ViT attention. No graph facts or bounding-box claims are inferred.

## Decode latent embeddings to text

```python
decode = vec.TextDecoder.from_foundation(
    './flan-t5-small', input_space=encode.output_space,
    local_files_only=True, bridge='linear',
    generation={'max_new_tokens': 32},
)
answer = decode(latent)
```

The linear bridge projects vectors into the foundation's input-embedding width.
The foundation encoder processes the resulting embedding sequence before its
language decoder generates text. Ordered `context={'latents': [...]}` prefixes
are projected in the same declared input space and participate in attention.

**The foundation is pretrained; a new linear bridge is not.** This call is a
working parameterized path, not evidence that an arbitrary encoder/decoder pair
already shares semantics. Train the bridge with appropriate paired targets.
Identity bridging requires the exact declared native *input-embedding* space;
final encoder states are a different space even when their dimensions match.
`decoder.embed_text(...)` exposes actual native input embeddings for baseline
comparisons, not final transformer encodings.

## Collect, train, save and reload

Generation and differentiable objectives are separate. Targets enter the language
loss, never the encoder's source sequence. This authored example trains the text
decoder from previously computed vectors:

```python
from tensorcode import training

trainer = training.ToolTrainer(decode, lr=0.0001)
experience = trainer.capture(
    latent, 'The database connection was restored.',
    source='authored-example:incident-review-17',
)
codecs = vec.latent_codecs()  # explicit trusted Space/Latent serialization types
experience.save('./experience.json', operations=trainer.operations,
                codecs=codecs, release=True)
loaded = training.load('./experience.json', operations=trainer.operations,
                       codecs=codecs)
trainer.fit([loaded], epochs=1)
decode.save_pretrained('./text-decoder')
trainer.save_checkpoint('./text-training', progress={'reviewed_examples': 1})
restored = vec.TextDecoder.from_pretrained('./text-decoder')
resumed = training.ToolTrainer(restored, lr=0.0001)
resumed.load_checkpoint('./text-training')
```

This single reviewed pair demonstrates the lifecycle, not generalization. The
encoder was called outside the captured objective and is not trained by this
example. To include decoder context in captured supervision, use
`{'value': latent, 'context': {'latents': [other_latent]}}` as the capture input.
Training dropout can resample during replay; exact continuation additionally
requires restored RNG and supported deterministic device operations.

## Decode latent conditioning to an image

```python
image_decode = vec.ImageDecode.from_foundation(
    './diffusion-foundation', input_space=conditioning.space,
    local_files_only=True, conditioning_projection='identity',
    num_inference_steps=4,
)
pixels = image_decode(conditioning, context={'seed': 17})
image_decode.save_pretrained('./image-decoder')
restored_image_decode = vec.ImageDecode.from_pretrained('./image-decoder')
```

Here `conditioning` must already contain that foundation's native conditioning
embeddings. Choosing `identity` explicitly declares compatibility, including
position and model semantics; matching widths does not prove it. The default
`conditioning_projection='linear'` instead owns a newly initialized adapter that
requires training. Context uses `{'latents': [...]}` in the declared input space.

The decoder owns a conditional UNet, VAE and **DDIM scheduler**. It projects the
conditioning, denoises latent noise, then decodes RGB pixels in `[0,1]`, shape
`[batch,3,height,width]`. Exactly one of `context['seed']` or `context['noise']` is
required. Sampling uses a local scheduler and restores model modes; unsupported
pipeline architectures are rejected. This is not a universal loader for every
Diffusers pipeline, and substituting DDIM can differ from a foundation's original
sampler. `ImageDecoder` and `ImageDecode` name the same operation.

For supervised learning, call `loss(value, target_pixels, noise=..., timesteps=...,
context=...)`. The VAE encodes target images for the diffusion loss; targets never
become text/latent conditioning. With `ToolTrainer`, capture inputs are
`{'value': conditioning, 'noise': noise, 'timesteps': timesteps, 'context': ...}`
and targets are RGB image tensors. Noise and timesteps are explicit saved inputs.
The same experience/checkpoint lifecycle and `vec.latent_codecs()` apply.

## Migration from 0.3

- `tensorcode.ops.llm` is now `tensorcode.ops.text`, including its image attachments
  and provider-neutral message operations. There is no old-namespace alias.
- The old vocabulary/mean embedding encoder is `vec.VocabularyEncoder`.
- The old convolution/supplied patch encoder is `vec.PatchEncoder`.
- `vec.TextEncoder` and `vec.ImageEncoder` now mean owned transformer operations.
- Generic supplied-module `vec.Decode` remains available. `vec.TextDecode` and
  `vec.ImageDecode` select the new owned decoders.

Old serialized experiences can contain the former operation identities. Recreate
or explicitly migrate their bindings; the loader does not silently accept a
mismatched configuration. Historical evaluation records retain their original
identities. Symbolic graph operations remain stubs.
