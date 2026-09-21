"""Train/evaluate an owned scene ranker on supplied image-and-candidate JSONL.

Each record: image_path, source_id, question, candidates:[{id,text}], target ID.
Images must be caller-licensed RGB images. Resizing is explicit preprocessing;
reported patch coordinates refer to that resized input, not the original photo.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import re

import torch
from PIL import Image

from tensorcode.tools.scene import Scene
from tensorcode.training import ToolTrainer


def read_records(path, size):
    records = []
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        image_path = Path(row['image_path'])
        if not image_path.is_absolute():
            image_path = Path(path).parent / image_path
        with Image.open(image_path) as image:
            image = image.convert('RGB').resize((size, size))
            pixels = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8).to(torch.float32).reshape(size, size, 3).permute(2, 0, 1) / 255
        records.append(({'pixels': pixels, 'question': row['question'], 'source_id': row['source_id'], 'candidates': row['candidates']}, row['target']))
    if not records:
        raise ValueError('dataset must not be empty')
    return records


@torch.no_grad()
def evaluate(tool, rows):
    result = {}
    for mode in ['full', 'blank_image', 'zero_workspace', 'bypass_workspace']:
        correct, loss = 0, 0.
        for inputs, target in rows:
            value = dict(inputs, pixels=torch.zeros_like(inputs['pixels'])) if mode == 'blank_image' else inputs
            ablation = {'zero_workspace': 'zero', 'bypass_workspace': 'bypass'}.get(mode)
            logits = tool.rank.compute(value, workspace_ablation=ablation)[0]
            index = [item['id'] for item in inputs['candidates']].index(target)
            correct += int(int(logits.argmax()) == index)
            loss += torch.nn.functional.cross_entropy(logits[None], torch.tensor([index], device=logits.device)).item()
        result[mode] = {'accuracy': correct / len(rows), 'cross_entropy': loss / len(rows), 'count': len(rows)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', help='Training JSONL; omit to evaluate a saved model')
    parser.add_argument('--test', required=True)
    parser.add_argument('--model', required=True, help='Output/local model directory or Hub ID for evaluation')
    parser.add_argument('--foundation', help='Explicit CLIP Hub foundation for a fresh trainable ranker')
    parser.add_argument('--revision', help='Pinned foundation revision')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--image-size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--report')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    report = {'seed': args.seed, 'image_size': args.image_size, 'foundation': args.foundation, 'foundation_revision': args.revision, 'epochs': args.epochs, 'limitation': 'Explicit candidate ranking; attention is not proof and this does not establish general scene understanding.'}
    test = read_records(args.test, args.image_size)
    if args.train:
        rows = read_records(args.train, args.image_size)
        train_sources = {x['source_id'] for x, _ in rows}
        if train_sources & {x['source_id'] for x, _ in test}:
            raise ValueError('training and evaluation image sources must be disjoint')
        vocabulary = sorted({word for value, _ in rows for text in [value['question']] + [x['text'] for x in value['candidates']] for word in re.findall(r'\w+|[^\w\s]', text.casefold())})
        if args.foundation:
            if not args.revision:
                parser.error('--foundation requires --revision')
            tool = Scene.from_foundation(args.foundation, revision=args.revision).to(args.device)
        else:
            tool = Scene({'vocabulary': vocabulary, 'dimensions': 32, 'slots': 4, 'steps': 2, 'max_image_size': args.image_size, 'patch_size': 8}).to(args.device)
        # All image/text/workspace/scoring parameters already exist here.
        trainer = ToolTrainer(tool, optimizer=lambda parameters: torch.optim.Adam(parameters, lr=.001))
        report['before'] = evaluate(tool, test)
        output = Path(args.model)
        output.mkdir(parents=True, exist_ok=True)
        # Demonstrate durable feedback collection separately from published weights.
        capture = trainer.capture(*rows[0], source=f'user-supplied:{args.train}')
        capture.save(output.parent / (output.name + '-experience.json'), operations=trainer.operations)
        report['losses'] = []
        for _ in range(args.epochs):
            random.shuffle(rows)
            losses = []
            for inputs, target in rows:
                trainer.optimizer.zero_grad()
                loss = tool.loss(inputs, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(tool.parameters(), 1.)
                trainer.optimizer.step()
                trainer.steps += 1
                losses.append(float(loss.detach()))
            report['losses'].append(sum(losses) / len(losses))
        tool.save_pretrained(output)
        trainer.save_checkpoint(output.parent / (output.name + '-training'), progress={'epochs': args.epochs})
        report['train_count'] = len(rows)
    tool = Scene.from_pretrained(args.model, device=args.device)
    report['model_configuration'] = tool.configuration()
    report['after_reload'] = evaluate(tool, test)
    text = json.dumps(report, indent=2)
    if args.report:
        Path(args.report).write_text(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
