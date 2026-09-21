"""Train a graph decision on official MUTAG with a fixed graph-disjoint split.

Pass the official archive; this example never downloads data. Atom and bond labels
are supplied dataset features, not chemical understanding inferred from images.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time
import zipfile

SOURCE = 'https://www.chrsmrrs.com/graphkerneldatasets/MUTAG.zip'
SHA256 = 'c419bdc853c367d2d83da4973c45100954ae15e10f5ae2cddde6ca431f8207f6'


def read(path):
    with zipfile.ZipFile(path) as archive:
        def integers(name):
            return [int(line) for line in archive.read(f'MUTAG/MUTAG_{name}.txt').decode().splitlines() if line.strip()]
        labels = integers('graph_labels')
        membership = integers('graph_indicator')
        nodes = integers('node_labels')
        bonds = integers('edge_labels')
        edges = [tuple(int(x.strip()) for x in line.split(','))
                 for line in archive.read('MUTAG/MUTAG_A.txt').decode().splitlines() if line.strip()]
    if len(nodes) != len(membership) or len(edges) != len(bonds):
        raise ValueError('MUTAG node or edge metadata length mismatch')
    graphs = [{'id': i + 1, 'label': label, 'node_labels': [], 'edges': []} for i, label in enumerate(labels)]
    local = {}
    for global_id, (graph_id, label) in enumerate(zip(membership, nodes), start=1):
        if not 1 <= graph_id <= len(graphs):
            raise ValueError('Invalid graph membership')
        graph = graphs[graph_id - 1]
        local[global_id] = len(graph['node_labels'])
        graph['node_labels'].append(label)
    for (a, b), bond in zip(edges, bonds):
        if a not in local or b not in local:
            raise ValueError('Edge references missing node')
        if membership[a - 1] != membership[b - 1]:
            raise ValueError('Edge crosses graph boundary')
        graphs[membership[a - 1] - 1]['edges'].append((local[a], local[b], bond))
    return graphs


def split_ids(count, *, seed=7):
    ids = list(range(count))
    random.Random(seed).shuffle(ids)
    boundary = int(count * .8)
    return ids[:boundary], ids[boundary:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--epochs', type=int, default=80)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('--epochs must be positive')
    digest = hashlib.sha256(args.data.read_bytes()).hexdigest()
    if digest != SHA256:
        parser.error('Archive does not match the pinned official MUTAG SHA256')

    import torch
    from torch import nn
    import tensorcode as tc
    from tensorcode.ops.graph import Graph
    from tensorcode.ops.graph.neural import GraphEncoder
    from tensorcode.ops.vec import Classify, Space

    torch.set_num_threads(2)
    torch.manual_seed(7)
    torch.use_deterministic_algorithms(True)
    records = read(args.data)
    train_ids, test_ids = split_ids(len(records), seed=7)
    # Categories are defined from training inputs only. Unseen test atom types get
    # their own unknown coordinate; held-out labels never configure the model.
    atom_types = sorted({atom for i in train_ids for atom in records[i]['node_labels']})
    atom_lookup = {atom: i for i, atom in enumerate(atom_types)}
    classes = sorted({records[i]['label'] for i in train_ids})
    class_lookup = {label: i for i, label in enumerate(classes)}
    dimensions = len(atom_types) + 1
    graphs = []
    for record in records:
        features = []
        for atom in record['node_labels']:
            feature = [0.] * dimensions
            feature[atom_lookup.get(atom, dimensions - 1)] = 1.
            features.append({'features': feature})
        nodes = tuple(str(i) for i in range(len(features)))
        graphs.append(Graph(
            nodes=nodes,
            edges=tuple((str(a), str(bond), str(b)) for a, b, bond in record['edges']),
            sources=(SOURCE,), identity=f"MUTAG:{record['id']}",
            node_attributes=features,
        ))
    space = Space('mutag-message-passing', 32)
    encoder = GraphEncoder(dimensions, 32, 32, space=space, steps=2)
    classifier = Classify(nn.Linear(32, len(classes)), labels=tuple(map(str, classes)))
    parameters = list(encoder.parameters()) + list(classifier.parameters())
    optimizer = torch.optim.Adam(parameters, lr=.01)

    def logits_for(ids):
        # Explicit authored pooling combines variable node counts, keeping all
        # gradients through the public trainable graph operation.
        pooled = torch.stack([encoder(graphs[i]).tensor.mean(dim=0) for i in ids])
        return classifier(pooled).logits

    def targets(ids):
        return torch.tensor([class_lookup[records[i]['label']] for i in ids])

    def evaluate(ids):
        encoder.eval()
        classifier.eval()
        with torch.no_grad():
            logits = logits_for(ids)
            expected = targets(ids)
            return {'accuracy': float((logits.argmax(-1) == expected).float().mean()),
                    'cross_entropy': float(nn.functional.cross_entropy(logits, expected))}

    majority = max(classes, key=lambda c: sum(records[i]['label'] == c for i in train_ids))
    baseline = sum(records[i]['label'] == majority for i in test_ids) / len(test_ids)
    before = evaluate(test_ids)
    initial = [parameter.detach().clone() for parameter in encoder.parameters()]
    rng = random.Random(7)
    started = time.perf_counter()
    for epoch in range(args.epochs):
        encoder.train()
        classifier.train()
        order = train_ids.copy()
        rng.shuffle(order)
        for start in range(0, len(order), 16):
            ids = order[start:start + 16]
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(logits_for(ids), targets(ids))
            loss.backward()
            optimizer.step()
        if epoch == 0 or (epoch + 1) % 20 == 0:
            print(f'epoch {epoch + 1}/{args.epochs}: final training batch loss {loss.item():.4f}', flush=True)
    elapsed = time.perf_counter() - started
    after = evaluate(test_ids)
    train_result = evaluate(train_ids)
    changed = any(not torch.equal(a, b) for a, b in zip(initial, encoder.parameters()))
    if not changed:
        raise RuntimeError('Graph encoder parameters did not learn')
    report = {
        'dataset': 'MUTAG official TU graph classification dataset',
        'source': SOURCE, 'sha256': digest,
        'graphs': len(graphs), 'train_graphs': len(train_ids), 'test_graphs': len(test_ids),
        'split': '80/20 graph-disjoint random shuffle, seed 7; no stratification',
        'train_graph_ids': [records[i]['id'] for i in train_ids],
        'test_graph_ids': [records[i]['id'] for i in test_ids],
        'seed': 7, 'epochs': args.epochs, 'batch_size': 16,
        'hidden_dimensions': 32, 'output_dimensions': 32, 'message_passing_steps': 2,
        'pooling': 'authored mean over nodes',
        'atom_types_from_training': atom_types, 'unknown_atom_coordinate': dimensions - 1,
        'optimizer': 'Adam', 'learning_rate': .01,
        'training_majority_label': majority, 'heldout_majority_accuracy': baseline,
        'before': before, 'after': after, 'training_after': train_result,
        'interpretation': 'Compare cross-entropy as well as accuracy against the random initialization; on this small split, one graph changes accuracy by 2.63 percentage points.',
        'graph_encoder_parameters_changed': changed,
        'training_seconds': elapsed, 'tensorcode': tc.__version__, 'torch': torch.__version__,
        'limitations': 'Single fixed small split, no confidence interval or benchmark ranking. Atom categories and graph structure are supplied dataset features. Message passing ignores bond types although graph edges preserve them. Hyperparameters fixed before held-out evaluation; no held-out tuning. This demonstrates supervised graph learning, not inferred chemistry or autonomous cognition.',
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + '\n')


if __name__ == '__main__':
    main()
