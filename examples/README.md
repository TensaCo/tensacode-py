# Runnable examples

Run these from the repository root after installing the relevant extras. Inputs,
model weights, checkpoints and generated reports belong outside the checkout.
Recorded evaluation results live in [docs/results](../docs/results/); these scripts
do not download data or weights automatically.

| Example | Demonstrates | Dependencies |
|---|---|---|
| [banking77_restart.py](banking77_restart.py) | Capture labeled traces, exit, reload/train, then evaluate a checkpoint in another process | `tensorcode[vec]` |
| [mutag.py](mutag.py) | Train a graph encoder on a fixed graph-disjoint molecule split | `tensorcode[vec]` |
| [local_multimodal.py](local_multimodal.py) | Evaluate supplied vision models on the published candy photograph, retaining invalid structured answers as failures | `tensorcode[local]` |

## Durable text learning

Obtain the official train/test CSVs from
[PolyAI Banking77](https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data).
Choose a new artifact directory for each run:

```bash
python examples/banking77_restart.py \
  --train /path/to/train.csv --test /path/to/test.csv \
  --artifacts /tmp/banking77-run --output /tmp/banking77-results.json
```

This is the canonical Banking77 example. It replaces the earlier, redundant
in-process training script. Supervision comes from supplied dataset labels.

## Graph learning

Obtain the [official MUTAG archive](https://www.chrsmrrs.com/graphkerneldatasets/MUTAG.zip):

```bash
python examples/mutag.py --data /path/to/MUTAG.zip \
  --output /tmp/mutag-results.json
```

The script verifies the archive hash and uses a fixed split. Atom categories and
adjacency are supplied dataset features, not inferred chemical knowledge.

## Multimodal smoke evaluation

Install `tensorcode[local]`. Explicitly download the model first, for example with
`hf download HuggingFaceTB/SmolVLM-256M-Instruct --revision 7e3e67edbbed1bf9888184d9df282b700a323964`.
Save the published
[candy photograph](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG)
locally, then run:

```bash
python examples/local_multimodal.py --image /path/to/candy.JPG \
  --source https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG \
  --output /tmp/multimodal-results.json
```

The prompts are specific to this image. This is a smoke test, not a general image
benchmark. Use `--device cuda` when available. To select another downloaded model,
supply **both** `--model` and its matching `--revision`; the defaults pin SmolVLM.
Model outputs can be wrong or fail JSON validation. These failures remain in the
report and must not be interpreted as successful decisions.

Each script supports `--help`. The package [README](../README.md) contains smaller
API examples; [documentation](../docs/README.md) explains contracts and limitations.
