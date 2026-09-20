"""Installed VerbNet alternatives for authored frames; no learned meaning claims."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time
from unittest.mock import patch

from eval.learning.evaluate_model_investigation import encode, sha
from tensorcode.language import Frame, verbnet
from tensorcode.records import Ref


def signature(proposal):
    return proposal.goal


def same_set(left, right):
    return len(left) == len(right) and all(any(verbnet._exact(signature(a), signature(b)) for b in right) for a in left)


def resource_manifest(root):
    return {p.name: sha(p) for p in sorted(root.glob('*.xml'))}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=Path('eval/results/goal_alternatives.json'))
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[2]
    files = [Path(__file__).resolve(), root/'eval/learning/evaluate_model_investigation.py',
             root/'src/tensorcode/language/verbnet.py', root/'src/tensorcode/language/semantics.py',
             root/'src/tensorcode/goals.py', root/'src/tensorcode/agent/core.py',
             root/'src/tensorcode/agent/goal_interpretation.py']
    hashes = {str(p): sha(p) for p in files}
    resources = verbnet.find_verbnet()
    if resources is None:
        raise RuntimeError('installed VerbNet is required; this evaluator does not download resources')
    manifest = resource_manifest(resources)
    lexicon = verbnet.load(resources)
    reversed_lexicon = {key: tuple(replace(vc, frames=tuple(reversed(vc.frames))) for vc in reversed(classes))
                        for key, classes in lexicon.items()}
    inputs = (Frame('make', {'object': Ref('audit:object')}, {'mood': 'imperative'}),
              Frame('move', {'object': Ref('audit:object'), 'destination': Ref('audit:destination')},
                    {'mood': 'imperative'}))
    rows = []
    for frame in inputs:
        started = time.perf_counter()
        original = verbnet.goal_candidates(frame, lexicon, max_derivations=100000)
        reordered = verbnet.goal_candidates(frame, reversed_lexicon, max_derivations=100000)
        calls = Counter()
        def prior(*args, **kwargs):
            calls['class_prior'] += 1
            return 1e100
        def counts(*args, **kwargs):
            calls['sense_counts'] += 1
            return {'audit:artificial-prior': 1e100}
        retired_helpers_absent = not hasattr(verbnet, 'class_prior') and not hasattr(verbnet, 'sense_counts')
        with patch.object(verbnet, 'class_prior', prior, create=True), patch.object(verbnet, 'sense_counts', counts, create=True):
            reprioritized = verbnet.goal_candidates(frame, lexicon, max_derivations=100000)
        limited = verbnet.goal_candidates(frame, lexicon, max_derivations=1)
        rows.append({'authored_frame': frame, 'lexical_classes': tuple(vc.id for vc in lexicon.get(frame.predicate, ())),
                     'complete_batch': original, 'reordered_batch': reordered,
                     'changed_prior_batch': reprioritized, 'limited_batch': limited,
                     'counts': {'proposals': len(original.proposals),
                                'derivations': sum(len(p.derivations) for p in original.proposals),
                                'clean_proposals': sum(not p.goal.unmapped_roles and any(not d.obligations for d in p.derivations)
                                                       for p in original.proposals)},
                     'checks': {'complete': original.complete and reordered.complete and reprioritized.complete,
                                'class_frame_order_same_semantic_set': same_set(original.proposals, reordered.proposals),
                                'changed_priors_same_semantic_set': same_set(original.proposals, reprioritized.proposals),
                                'priors_not_consulted': not sum(calls.values()),
                                'retired_prior_helpers_absent': retired_helpers_absent,
                                'small_budget_explicitly_incomplete': not limited.complete and bool(limited.unresolved)},
                     'prior_calls': dict(calls), 'elapsed_ms': (time.perf_counter()-started)*1000})
    report = {'evaluation': 'installed_verbnet_authored_frame_goal_alternatives',
              'source_hashes': hashes, 'source_hashes_verified_after_run': {p: sha(p)==h for p,h in hashes.items()},
              'resource_root': str(resources), 'resource_xml_manifest': manifest,
              'resource_manifest_sha256': hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
              'resource_hashes_verified_after_run': resource_manifest(resources) == manifest,
              'max_derivations_complete_audit': 100000, 'max_derivations_truncated_audit': 1,
              'cases': rows, 'limitations': ['Frames and references supplied, not learned from language.',
                  'Inventory and role/result projection authored; SYNRESTRS lost in upstream loader.',
                  'Semantic sets compare full typed Goals including aggregate class labels; derivation frame indices excluded.',
                  'No candidate selection, execution, semantic accuracy, or intent inference measured.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(encode(report), indent=2, allow_nan=False)+'\n')
    print(args.output)


if __name__ == '__main__':
    main()
