"""Evaluate supplied local models on the published candy photograph (not a benchmark).

Download the model explicitly with `hf download MODEL_ID`, install tensorcode[local],
and supply a local image file. This script never downloads images or model weights.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
from tensorcode import trace
from tensorcode.integrations.local import LocalModel
from tensorcode.ops import text as text_ops


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--model', default='HuggingFaceTB/SmolVLM-256M-Instruct')
    parser.add_argument('--revision', default='7e3e67edbbed1bf9888184d9df282b700a323964')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    raw = args.image.read_bytes()
    model = LocalModel.from_pretrained(args.model, revision=args.revision,
                                       device=args.device, max_new_tokens=160)
    image = text_ops.ImageEncoder(media_type='image/jpeg', source_ref=args.source)(raw)[0].content[0]
    records = []
    questions = [
        'Describe what the hand is holding.',
        'How many candies are in the hand?',
        'What animal is drawn on the candy?',
    ]
    for question in questions:
        value = (text_ops.Message('user', (image, text_ops.TextPart(question))),)
        started = time.monotonic()
        try:
            with trace() as session:
                response = text_ops.Transform(model)(value)
            records.append({'task': question, 'answer': response[-1].content,
                            'seconds': time.monotonic() - started, 'trace_calls': len(session.calls)})
        except (ValueError, TypeError) as error:
            records.append({'task': question, 'status': 'rejected', 'error': str(error),
                            'seconds': time.monotonic() - started})
    # These deliberately exercise strict structured operations with a small model.
    # Failure is a measured outcome, not replaced with a made-up answer.
    structured = [
        ('classify', text_ops.Classify(model, labels=('food', 'vehicle'),
            instructions='Classify the pictured objects. Return JSON. Use null for unknown confidence and distribution.')),
        ('retrieve', text_ops.Retrieve(model, items={'food': 'Candy and other sweets', 'vehicle': 'Cars and trucks'},
            instructions='Find the item describing the pictured objects. Return JSON. Use null for scores.')),
    ]
    for task, operation in structured:
        started = time.monotonic()
        try:
            result = operation((text_ops.Message('user', (image, text_ops.TextPart('Identify the best match.'))),))
            records.append({'task': task, 'value': getattr(result, 'value', None),
                            'keys': list(getattr(result, 'keys', ())),
                            'status': 'valid', 'seconds': time.monotonic() - started})
        except (ValueError, TypeError) as error:
            records.append({'task': task, 'status': 'rejected', 'error': str(error),
                            'seconds': time.monotonic() - started})
    report = {'model_id': args.model, 'revision': args.revision,
              'image_source': args.source, 'image_sha256': hashlib.sha256(raw).hexdigest(),
              'torch': torch.__version__, 'device': args.device, 'records': records,
              'limitations': 'Single published image, not held-out benchmark. Semantics come from the supplied pretrained model. No training, calibrated confidence, or general visual understanding claim. Structured failures are retained.'}
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
