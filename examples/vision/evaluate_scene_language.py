"""Evaluate owned scene interpretations with real/blank/shuffled image evidence.

Input JSONL: image_path, source_id, question (spatial caption), target ('0'/'1').
VSR labels evaluate explicit yes/no judgments. Free descriptions are preserved for
review, not automatically scored as factual. No training occurs in this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import torch
from PIL import Image

from tensorcode.tools.scene import Scene


def read_rows(path, limit):
    result = []
    for line in Path(path).read_text().splitlines()[:limit]:
        row = json.loads(line)
        location = Path(row['image_path'])
        if not location.is_absolute():
            location = Path(path).parent / location
        with Image.open(location) as image:
            image = image.convert('RGB')
            image.thumbnail((1024, 1024))
            pixels = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8).reshape(image.height, image.width, 3).permute(2, 0, 1).float() / 255
        result.append((row, pixels))
    if len(result) < 2:
        raise ValueError('at least two images are required for shuffle evaluation')
    return result


def evaluate(model, rows, *, max_new_tokens):
    fingerprints = [hashlib.sha256(pixels.contiguous().view(torch.uint8).numpy().tobytes() + str(tuple(pixels.shape)).encode()).hexdigest() for _, pixels in rows]
    if len(set(fingerprints)) < 2:
        raise ValueError('shuffle evaluation requires distinct image contents')
    report = {'judgments': [], 'descriptions': [], 'metrics': {}, 'limitations': ['Descriptions are unverified model proposals, not extracted facts.', 'Foundation pretraining overlap with these images is unknown.', 'Blank/shuffle judgments are compared with original image labels; this is label retention, not altered-image ground-truth accuracy.', 'An inactive workspace residual preserves foundation behavior and establishes no learned TensorCode workspace benefit.']}
    for mode in ['full', 'blank', 'shuffle']:
        correct = parsed = 0
        for index, (row, pixels) in enumerate(rows):
            source_id = row['source_id']
            if mode == 'blank':
                pixels, source_id = torch.zeros_like(pixels), 'blank:' + source_id
            elif mode == 'shuffle':
                replacement = next((index + offset) % len(rows) for offset in range(1, len(rows)) if fingerprints[(index + offset) % len(rows)] != fingerprints[index])
                other, pixels = rows[replacement]
                source_id = other['source_id']
            question = 'Does the image support this description? Answer yes or no, then explain the spatial relationship: ' + row['question']
            receipt = model.interpret({'pixels': pixels, 'source_id': source_id, 'question': question}, max_new_tokens=max_new_tokens)
            match = re.match(r'^\s*(yes|no)\b', receipt['interpretation'], flags=re.I)
            prediction = None if match is None else str(int(match.group(1).lower() == 'yes'))
            parsed += prediction is not None
            correct += prediction == str(row['target'])
            report['judgments'].append({'mode': mode, 'original_source_id': row['source_id'], 'caption': row['question'], 'target': row['target'], 'prediction': prediction, 'receipt': receipt})
            if index < 4:
                receipt = model.interpret({'pixels': pixels, 'source_id': source_id, 'question': 'Describe the overall scene, spatial arrangement, and interactions. Mention uncertainty where details are unclear.'}, max_new_tokens=max_new_tokens)
                report['descriptions'].append({'mode': mode, 'original_source_id': row['source_id'], 'receipt': receipt})
        report['metrics'][mode] = {'count': len(rows), 'accuracy' if mode == 'full' else 'original_label_agreement': correct / len(rows), 'parsed_fraction': parsed / len(rows)}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--model', required=True, help='Owned Scene directory; output when importing a foundation')
    parser.add_argument('--foundation')
    parser.add_argument('--revision')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--limit', type=int, default=32)
    parser.add_argument('--max-new-tokens', type=int, default=64)
    parser.add_argument('--report', required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(23)
    if args.foundation:
        if not args.revision:
            parser.error('--foundation requires --revision')
        model = Scene.from_language_foundation(args.foundation, revision=args.revision)
        model.save_pretrained(args.model)
        del model
    model = Scene.from_pretrained(args.model, device=args.device)
    rows = read_rows(args.data, args.limit)
    with torch.inference_mode():
        report = evaluate(model, rows, max_new_tokens=args.max_new_tokens)
    report['model_config'] = model.configuration()
    report['data'] = str(Path(args.data).resolve())
    Path(args.report).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['metrics'], indent=2))


if __name__ == '__main__':
    main()
