"""Offline owned-operation lifecycle; two authored cases do not measure quality."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tensorcode import trace, training
from tensorcode.ops.vec import Classify, latent_codecs
from tensorcode.ops.vec.encode import VocabularyEncoder


def run(output: Path, *, epochs: int = 5):
    torch.manual_seed(7)
    output.mkdir(parents=True, exist_ok=False)
    space = {'name': 'example.reviewed-text', 'dimensions': 8}
    # Every parameter exists before collection and optimizer construction.
    operations = {
        'evidence': VocabularyEncoder({
            'vocabulary': ['database', 'network', 'refused', 'loss'],
            'dimensions': 8, 'output_space': space,
        }),
        'interpretation': Classify({
            'architecture': 'linear', 'input_space': space,
            'labels': ['database', 'network'],
        }),
    }
    texts = ['database refused', 'network loss']
    targets = ['database', 'network']

    def predict(bound):
        return bound['interpretation'](bound['evidence'](texts))

    with trace() as session:
        prediction = predict(operations)
    session.supervise(prediction, targets, source='two authored lifecycle cases')
    codecs = latent_codecs()
    session.save(output / 'experience.json', operations=operations, codecs=codecs, release=True)
    experience = training.load(output / 'experience.json', operations=operations, codecs=codecs)
    trainer = training.Trainer(operations, lr=0.05)
    losses = trainer.fit([experience], epochs=epochs)
    training.save_checkpoint(output / 'training.json', operations=operations, optimizer=trainer.optimizer)
    for name, operation in operations.items():
        operation.eval()
        operation.save_pretrained(output / name)
    restored = {
        'evidence': VocabularyEncoder.from_pretrained(output / 'evidence'),
        'interpretation': Classify.from_pretrained(output / 'interpretation'),
    }
    for operation in restored.values():
        operation.eval()
    with torch.no_grad():
        torch.testing.assert_close(predict(restored).logits, predict(operations).logits, rtol=0, atol=0)
    resumed = training.Trainer(restored, lr=0.05)
    training.load_checkpoint(output / 'training.json', operations=restored, optimizer=resumed.optimizer)
    replay = training.load(output / 'experience.json', operations=restored, codecs=codecs)
    resumed.step(replay)
    report = {'updates': len(losses), 'reload_equal': True, 'resume_update': True,
              'limitations': 'Authored lifecycle fixture; no held-out quality evaluation.'}
    print(report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=5)
    args = parser.parse_args()
    run(args.output, epochs=args.epochs)
