"""Compile algorithm relations into a real pretrained CLIP attention layer.

Uses a local Hugging Face checkpoint by default; no training or downloads occur.
This demonstrates insertion and gradient flow, NOT learned algorithm execution.
"""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from tensorcode.algorithm import compile
from tensorcode.algorithm.structure import BiasedMultiheadAttention


def main():
    from transformers import AutoImageProcessor, CLIPVisionModel

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="openai/clip-vit-base-patch32")
    parser.add_argument(
        "--revision", default="3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    model = CLIPVisionModel.from_pretrained(
        args.model,
        revision=args.revision,
        local_files_only=True,
        attn_implementation="eager",
    ).eval()
    processor = AutoImageProcessor.from_pretrained(
        args.model, revision=args.revision, local_files_only=True
    )
    image = Image.open(args.image).convert("RGB")
    pixels = processor(images=image, return_tensors="pt")["pixel_values"]
    with torch.no_grad():
        states = model(pixel_values=pixels, output_hidden_states=True).hidden_states[0]
        layer = model.encoder.layers[0]
        x = layer.layer_norm1(states).detach()
        expected = layer.self_attn(x)[0]
    algorithm = compile(
        """def policy(position, target, velocity):
 return 30 * (target - position) - 2.5 * velocity
""",
        inputs={"position": (20,), "target": (20,), "velocity": (20,)},
        dtype="float32",
    )
    native = layer.self_attn
    adapter = BiasedMultiheadAttention.from_projections(
        native.q_proj,
        native.k_proj,
        native.v_proj,
        native.out_proj,
        algorithm,
        heads=native.num_heads,
    ).eval()
    actual = adapter(x, x, x)[0]
    zero_error = (actual - expected).abs().max().item()
    if not torch.allclose(actual, expected, atol=3e-5, rtol=3e-5):
        raise RuntimeError("pretrained function preservation failed")
    actual.square().mean().backward()
    gradient = adapter.structure.strengths.grad.abs().sum().item()
    with torch.no_grad():
        adapter.structure.strengths.fill_(2.0)
        changed = adapter(x, x, x)[0]
    report = {
        "model": args.model,
        "revision": args.revision,
        "input_image": str(args.image),
        "tokens": x.shape[1],
        "zero_bias_max_error": zero_error,
        "relation_strength_gradient_l1": gradient,
        "nonzero_bias_output_change": (changed - expected).abs().max().item(),
        "frozen_projection_weights": True,
        "algorithm_nodes": len(algorithm.graph()["nodes"]),
        "binding": "untrained learned projections",
        "claim": "pretrained insertion and gradients only; no learned algorithm execution",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
