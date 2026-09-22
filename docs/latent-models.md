# Pretrained vector encoders and decoders

TensorCode provides owned transformer encoders and text/image generation from
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
from tensorcode.ops.vec.encode import TextEncoder, ImageEncoder
from tensorcode.ops.vec.decode import TextDecoder, ImageDecoder

encode = TextEncoder.from_foundation(
    './flan-t5-small',
    output_space=vec.Space('my-t5/states', 512, organization='sequence'),
    local_files_only=True, readout='sequence',
)
latent = encode('The service recovered after reconnecting the database.')
print(latent.tensor.shape, latent.space, latent.mask)
encode.save_pretrained('./text-encoder')
restored = TextEncoder.from_pretrained('./text-encoder')
```

Text encoding returns final native transformer states, with a batch axis and a
boolean token mask. `readout='pooled'` instead returns a masked mean in a feature
space. Optional context is an ordered `{'latents': [...]}` prefix in an explicit
`context_space` matching the native input-embedding width. Prefix vectors and their
masks participate in transformer attention; raw text context is not accepted.
With prefixes, valid tokens are packed in order before transformer processing,
so masked prefix padding cannot shift the primary text positions. Sequence outputs
retain the original primary-text layout and mask, with masked output slots zeroed.
The optional
`readout='output_encoding'` appends an owned trainable token after valid context
and input embeddings and returns its final transformer state in a feature space.
Text padding is compacted before appending the token, so its position does not
depend on other batch members. Inputs exceeding native position capacity are
rejected. The new token starts untrained even when native weights are pretrained;
none of these readouts establishes a shared semantic space by itself.

`ImageEncoder` currently supports **ViT**. It owns the model and processor.
Supply a sequence `Space` matching the ViT hidden width, or use `readout='pooled'`
with a feature space for the native final CLS state.
`readout='output_encoding'` instead appends a trainable token after context and
native image embeddings and reads its final state through the same ViT layers:

```python
vision = ImageEncoder.from_foundation(
    './vit-base', output_space=vec.Space('my-vit/patches', 768, organization='sequence'),
    readout='sequence', local_files_only=True,
)
patches = vision(vision.preprocess(images))  # supplied PIL image or image batch
```

Raw CHW/BCHW floating tensors in `[0,1]` must already match the configured image
size; their normalization remains differentiable. `preprocess` performs the saved
processor's resize/normalization and makes no input-gradient promise. Patch
coordinates refer to the **processed image**, not invented original-image regions.
Optional `context_space` and `context={'latents': [...]}` prefix compatible vectors
before ViT attention; the space must match its native embedding width. No graph
facts or bounding-box claims are inferred.

## Decode latent embeddings to text

```python
decode = TextDecoder.from_foundation(
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
Valid prefix and primary vectors are packed in order before projection; masked
padding does not introduce positional gaps or affect valid-input gradients.

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

trainer = training.Trainer.from_tool(decode, lr=0.0001)
experience = trainer.capture(
    latent, 'The database connection was restored.',
    source='authored-example:incident-review-17',
)
codecs = vec.latent_codecs()  # explicit trusted Space/Latent serialization types
experience.save('./experience.json', operations=trainer.operations,
                codecs=codecs, release=True)
loaded = training.load_experience('./experience.json', operations=trainer.operations,
                       codecs=codecs)
trainer.fit([loaded], epochs=1)
decode.save_pretrained('./text-decoder')
trainer.save_checkpoint('./text-training', progress={'reviewed_examples': 1})
restored = TextDecoder.from_pretrained('./text-decoder')
resumed = training.Trainer.from_tool(restored, lr=0.0001)
resumed.load_checkpoint('./text-training')
```

This single reviewed pair demonstrates the lifecycle, not generalization. The
encoder was called outside the captured objective and is not trained by this
example. To train the appended encoder readout together with the decoder bridge, see
[the connected readout example](../examples/output_encoding_learning.py): it
captures encoding inside the trace so replay reaches the encoder parameters.
To include decoder context in captured supervision, use
`{'value': latent, 'context': {'latents': [other_latent]}}` as the capture input.
Training dropout can resample during replay; exact continuation additionally
requires restored RNG and supported deterministic device operations.

## Decode latent conditioning to an image

```python
image_decode = ImageDecoder.from_foundation(
    './diffusion-foundation', input_space=conditioning.space,
    local_files_only=True, bridge='identity',
    num_inference_steps=4,
)
pixels = image_decode(conditioning, context={'seed': 17})
image_decode.save_pretrained('./image-decoder')
restored_image_decode = ImageDecoder.from_pretrained('./image-decoder')
```

Here `conditioning` must already contain that foundation's native conditioning
embeddings. Choosing `identity` explicitly declares compatibility, including
position and model semantics; matching widths does not prove it. The default
`bridge='linear'` instead owns a newly initialized adapter that
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
become text/latent conditioning. With `Trainer.from_tool`, capture inputs are
`{'value': conditioning, 'noise': noise, 'timesteps': timesteps, 'context': ...}`
and targets are RGB image tensors. Noise and timesteps are explicit saved inputs.
The same experience/checkpoint lifecycle and `vec.latent_codecs()` apply.

## Public paths and owned configuration

Concrete classes live in `tensorcode.ops.vec.encode` (`TextEncoder`,
`ImageEncoder`, `VocabularyEncoder`, `PatchEncoder`) and
`tensorcode.ops.vec.decode` (`Decode`, `TextDecoder`, `ImageDecoder`). Root exports
refer to the same classes. Backend modules remain private; artifact identities
use public operation paths.

Public learned constructors take JSON configuration and own their parameters.
`VocabularyEncoder` and `PatchEncoder` provide lightweight trainable mechanisms
without pretrained semantics. General vector operations provide owned linear/MLP
architectures and supported native transformers. Use explicit `from_module`
factories when integrating supplied vector modules. Unsupported arbitrary-module
artifact saves fail because configuration alone cannot reconstruct executable code.

Encoder vector sides use `output_space`; decoder vector sides use `input_space`.
Encoder readout is `sequence`, `pooled`, or `output_encoding`; pooled text uses
masked mean and pooled vision uses native CLS. The appended readout token is
created at initialization, receives gradients, and is saved with the operation. Decoder bridges are `linear` or `identity`.
Encoder latent-prefix conditioning declares `context_space` explicitly.

The current API has no legacy constructor or namespace compatibility paths.
Artifacts must match the current class/configuration contracts. Historical
measurements in [results](results/README.md) remain records of their original
implementations; they do not measure the new owned operation configurations.
Symbolic graph operations remain unimplemented.
