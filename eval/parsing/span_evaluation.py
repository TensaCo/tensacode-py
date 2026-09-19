"""Evaluate reader proposals against original CoNLL-U text with character spans.

No model training, downloads, or semantic selection. Gold annotations choose whole
candidate trees only for oracle metrics; these are unavailable to the real agent.
Run a small local smoke measurement with ``python -m eval.parsing.span_evaluation``.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import statistics
import time

Span = tuple[int, int]


@dataclass(frozen=True)
class Word:
    id: int
    form: str
    upos: str
    head: int
    label: str


@dataclass(frozen=True)
class Gold:
    sent_id: str
    text: str
    words: tuple[Word, ...]
    spans: dict[int, Span]
    empty_nodes: int = 0
    error: str | None = None


@dataclass(frozen=True)
class Arc:
    span: Span
    head: Span | None  # None is ROOT, never an unknown head.
    label: str


@dataclass(frozen=True)
class Group:
    span: Span
    candidates: tuple[tuple[Arc, ...], ...]


def parse_record(block: str) -> Gold:
    """Keep original text and align surface/MWT forms without normalization."""
    metadata = {}
    rows = []
    words = []
    spans = {}
    empty_nodes = 0
    try:
        for line in block.splitlines():
            if line.startswith('#'):
                if line.startswith('# ') and ' = ' in line:
                    key, value = line[2:].split(' = ', 1)
                    metadata[key] = value
                continue
            if not line.strip():
                continue
            row = line.split('\t')
            if len(row) != 10:
                raise ValueError('CoNLL-U row must have ten columns')
            rows.append(row)
            if '.' in row[0]:
                empty_nodes += 1
            elif row[0].isdigit():
                words.append(Word(int(row[0]), row[1], row[3], int(row[6]), row[7]))
        if 'text' not in metadata:
            raise ValueError('missing original # text')
        text = metadata['text']
        if [w.id for w in words] != list(range(1, len(words) + 1)):
            raise ValueError('basic word IDs must be consecutive and unique')
        if any(w.head not in {0, *(t.id for t in words)} for w in words):
            raise ValueError('basic dependency head is absent')
        cursor = 0
        covered_until = 0
        next_word = 1
        by_id = {w.id: w for w in words}
        for row in rows:
            if '.' in row[0]:
                continue
            if '-' in row[0]:
                lo, hi = map(int, row[0].split('-'))
                if lo >= hi or lo != next_word or lo <= covered_until or any(i not in by_id for i in range(lo, hi + 1)):
                    raise ValueError('invalid or overlapping multiword range')
                ids = range(lo, hi + 1)
                if ''.join(by_id[i].form for i in ids) != row[1]:
                    raise ValueError('non-concatenative multiword form has no exact child spans')
                covered_until = hi
            else:
                i = int(row[0])
                if i <= covered_until:
                    continue
                if i != next_word:
                    raise ValueError('surface rows do not follow basic word order')
                ids = (i,)
            next_word = max(ids) + 1
            while cursor < len(text) and text[cursor].isspace():
                cursor += 1
            if not text.startswith(row[1], cursor):
                raise ValueError(f'surface form does not match original text at character {cursor}')
            child_cursor = cursor
            for i in ids:
                spans[i] = (child_cursor, child_cursor + len(by_id[i].form))
                child_cursor = spans[i][1]
            cursor += len(row[1])
        if text[cursor:].strip() or len(spans) != len(words):
            raise ValueError('surface alignment did not cover the complete source')
        return Gold(metadata.get('sent_id', ''), text, tuple(words), spans, empty_nodes)
    except (ValueError, KeyError) as exc:
        return Gold(metadata.get('sent_id', ''), metadata.get('text', ''), tuple(words), {},
                    empty_nodes, str(exc))


def load_records(path: Path) -> tuple[Gold, ...]:
    return tuple(parse_record(block) for block in path.read_text('utf-8').strip().split('\n\n') if block.strip())


def gold_arcs(gold: Gold) -> tuple[Arc, ...]:
    if gold.error:
        raise ValueError(gold.error)
    return tuple(Arc(gold.spans[w.id], gold.spans[w.head] if w.head else None, w.label) for w in gold.words)


def reader_groups(text: str, readings) -> tuple[tuple[Group, ...], tuple[str, ...]]:
    """Validate reader source anchors, retaining invalid-candidate diagnostics."""
    groups = []
    errors = []
    for reading in readings:
        candidates = []
        group_span = None
        for alternative in reading.alternatives:
            metadata = alternative.metadata
            if not metadata.get('syntax_complete'):
                continue
            try:
                span = tuple(metadata['sentence_span'])
                if len(span) != 2 or any(type(v) is not int for v in span) or not (0 <= span[0] < span[1] <= len(text)):
                    raise ValueError('invalid sentence span')
                if text[span[0]:span[1]] != reading.text:
                    raise ValueError('sentence span does not match source')
                if group_span is not None and span != group_span:
                    raise ValueError('alternatives disagree about sentence span')
                anchors = {}
                previous_end = span[0]
                for anchor in metadata['token_anchors']:
                    lo, hi = anchor['char_span']
                    index = anchor['index']
                    if (type(index) is not int or type(lo) is not int or type(hi) is not int
                            or index != len(anchors) + 1 or not (previous_end <= lo < hi <= span[1])):
                        raise ValueError('duplicate or overlapping token anchor')
                    if (text[lo:hi] != anchor['token'] or index > len(reading.tokens)
                            or reading.tokens[index - 1] != anchor['token']):
                        raise ValueError('token anchor does not match source')
                    anchors[index] = (lo, hi)
                    previous_end = hi
                heads, labels = metadata['heads'], metadata['labels']
                if (not anchors or len(anchors) != len(reading.tokens) or set(heads) != set(anchors)
                        or set(labels) != set(anchors) or any(type(k) is not int for k in heads)
                        or any(type(k) is not int for k in labels)):
                    raise ValueError('dependency keys do not cover anchored tokens')
                if any(type(h) is not int or (h not in anchors and h != 0) for h in heads.values()):
                    raise ValueError('dependency head has no source anchor')
                if any(not isinstance(label, str) or not label for label in labels.values()):
                    raise ValueError('dependency labels must be nonempty strings')
                for index in anchors:
                    seen = set()
                    node = index
                    while node:
                        if node in seen:
                            raise ValueError('cyclic dependency candidate')
                        seen.add(node)
                        node = heads[node]
                if sum(h == 0 for h in heads.values()) != 1:
                    raise ValueError('candidate must have one root')
                arcs = tuple(Arc(anchors[i], anchors[heads[i]] if heads[i] else None, labels[i]) for i in anchors)
                if arcs not in candidates:
                    candidates.append(arcs)
                group_span = span
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(f'{type(exc).__name__}: {exc}')
        if group_span is not None:
            groups.append(Group(group_span, tuple(candidates)))
    ordered = sorted(groups, key=lambda g: g.span)
    if any(a.span[1] > b.span[0] for a, b in zip(ordered, ordered[1:])):
        return (), tuple(errors + ['overlapping sentence groups cannot be scored independently'])
    return tuple(ordered), tuple(errors)


def score(gold: Gold, groups: tuple[Group, ...]) -> dict:
    """Score trusted, validated Groups; reader_groups is the production-data boundary.

    Oracle selection always chooses whole candidates, never individual arcs.
    """
    denominator = sum(w.upos != 'PUNCT' for w in gold.words)
    result = {'gold_nonpunct_words': denominator, 'gold_words': len(gold.words),
              'alignment_error': gold.error, 'first': {}, 'oracle': {}}
    if gold.error:
        groups = ()
        targets, all_targets = {}, {}
    else:
        all_targets = {a.span: a for a in gold_arcs(gold)}
        targets = {gold.spans[w.id]: all_targets[gold.spans[w.id]] for w in gold.words if w.upos != 'PUNCT'}
    def candidate_counts(candidate):
        predicted = {a.span: a for a in candidate}
        uas = las = conditional = 0
        for span, target in targets.items():
            got = predicted.get(span)
            if got is None:
                continue
            # Both endpoints are representable, regardless of which head was chosen.
            if target.head is None or target.head in predicted:
                conditional += 1
            if got.head == target.head:
                uas += 1
                las += got.label == target.label
        return {'uas_correct': uas, 'las_correct': las, 'conditional_words': conditional,
                'aligned_words': sum(span in all_targets for span in predicted), 'predicted_words': len(predicted)}
    first = tuple(arc for group in groups for arc in (group.candidates[0] if group.candidates else ()))
    result['first'] = candidate_counts(first)
    # Choose one coherent tree per disjoint predicted sentence, independently for UAS/LAS.
    ua = []
    la = []
    for group in groups:
        if group.candidates:
            ua.extend(max(group.candidates, key=lambda c: candidate_counts(c)['uas_correct']))
            la.extend(max(group.candidates, key=lambda c: candidate_counts(c)['las_correct']))
    result['oracle'] = candidate_counts(tuple(la))
    result['oracle']['uas_correct'] = candidate_counts(tuple(ua))['uas_correct']
    result['oracle']['uas_conditional_words'] = candidate_counts(tuple(ua))['conditional_words']
    result['oracle']['exact_tree'] = bool(not gold.error and len(groups) == 1 and
        any(len(c) == len(all_targets) and all(all_targets.get(a.span) == a for a in c)
            for c in groups[0].candidates))
    result['first']['exact_tree'] = bool(not gold.error and len(groups) == 1 and
        len(first) == len(all_targets) and all(all_targets.get(a.span) == a for a in first))
    result['candidate_groups'] = len(groups)
    result['candidate_count'] = sum(len(g.candidates) for g in groups)
    return result


def summarize(rows: list[dict]) -> dict:
    total = sum(r['gold_nonpunct_words'] for r in rows)
    words = sum(r['gold_words'] for r in rows)
    out = {'sentences': len(rows), 'gold_nonpunct_words': total, 'gold_words': words,
           'dataset_alignment_errors': sum(bool(r['alignment_error']) for r in rows),
           'reader_errors': sum(bool(r.get('reader_error')) for r in rows),
           'invalid_candidate_inputs': sum(bool(r.get('candidate_errors')) for r in rows)}
    def ratio(num, den):
        return num / den if den else None
    for method in ('first', 'oracle'):
        counts = {k: sum(r[method][k] for r in rows) for k in
                  ('uas_correct', 'las_correct', 'conditional_words', 'aligned_words', 'predicted_words')}
        uas_conditional = sum(r[method].get('uas_conditional_words', r[method]['conditional_words']) for r in rows)
        out[method] = {**counts, 'uas': ratio(counts['uas_correct'], total),
                       'las': ratio(counts['las_correct'], total),
                       'word_alignment_precision': ratio(counts['aligned_words'], counts['predicted_words']),
                       'word_alignment_recall': ratio(counts['aligned_words'], words),
                       'conditional_uas_denominator': uas_conditional,
                       'conditional_uas': ratio(counts['uas_correct'], uas_conditional),
                       'conditional_las': ratio(counts['las_correct'], counts['conditional_words']),
                       'exact_tree_recall': ratio(sum(r[method]['exact_tree'] for r in rows), len(rows))}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--treebank', type=Path, default=Path.home() / '.cache/tensorcode/seeds/UD_English-EWT/en_ewt-ud-test.conllu')
    ap.add_argument('--model', type=Path, default=Path.home() / '.cache/tensorcode/models/ud_ewt_parser.pickle')
    ap.add_argument('--sample', type=int, default=12)
    ap.add_argument('--seed', type=int, default=20260921)
    ap.add_argument('--max-tokens', type=int, default=20)
    ap.add_argument('--output', type=Path, default=Path('eval/results/parsing_spans_smoke.json'))
    args = ap.parse_args()
    if args.sample < 1 or args.max_tokens < 1:
        ap.error('sample and max-tokens must be positive')
    from tensorcode.agent.understand import LearnedReader
    import tensorcode.agent.understand as understand_module
    import tensorcode.language.learned_parser as parser_module
    import tensorcode.language.deps_semantics as semantic_module
    import tensorcode.language.chart as chart_module
    import tensorcode.language.treebank as treebank_module
    paths = (Path(__file__), Path(understand_module.__file__), Path(parser_module.__file__), Path(semantic_module.__file__),
             Path(chart_module.__file__), Path(treebank_module.__file__))
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    records = load_records(args.treebank)
    eligible = [r for r in records if r.error or 2 <= len(r.words) <= args.max_tokens]
    chosen = random.Random(args.seed).sample(eligible, min(args.sample, len(eligible)))
    if not chosen:
        ap.error('no eligible local records')
    model_hash = hashlib.sha256(args.model.read_bytes()).hexdigest()
    dataset_hash = hashlib.sha256(args.treebank.read_bytes()).hexdigest()
    reader = LearnedReader(args.model)
    rows = []
    for record in chosen:
        start = time.perf_counter()
        error = None
        readings = []
        groups, errors = (), ()
        if not record.error:
            try:
                readings = reader.read(record.text)
                groups, errors = reader_groups(record.text, readings)
            except Exception as exc:
                error = f'{type(exc).__name__}: {exc}'
        measured = score(record, groups)
        rows.append({**measured, 'sent_id': record.sent_id, 'reader_error': error,
                     'candidate_errors': errors, 'empty_nodes_excluded': record.empty_nodes,
                     'syntax_candidates': measured['candidate_count'],
                     'semantic_candidates': sum(len(s.alternatives) for s in readings),
                     'proposals_with_acts': sum(bool(a.acts) for s in readings for a in s.alternatives),
                     'retention_discarded': sum(max((a.metadata.get('proposals_discarded', 0) for a in s.alternatives), default=0) for s in readings),
                     'search_truncated': any(a.metadata.get('search_truncated', False) for s in readings for a in s.alternatives),
                     'latency_ms': (time.perf_counter() - start) * 1000})
    result = {'evaluation': 'original_text_span_aligned_reader', 'sources': hashes,
              'model_sha256': model_hash,
              'source_hashes_verified_after_run': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() == h for p, h in ((Path(p), h) for p, h in hashes.items())},
              'model_hash_verified_after_run': hashlib.sha256(args.model.read_bytes()).hexdigest() == model_hash,
              'dataset_hash_verified_after_run': hashlib.sha256(args.treebank.read_bytes()).hexdigest() == dataset_hash,
              'dataset': {'path': str(args.treebank), 'sha256': dataset_hash,
                          'source': 'exact # text; surface/MWT forms provide unnormalized character spans',
                          'records': len(records), 'eligible': len(eligible), 'min_words': 2,
                          'max_words': args.max_tokens, 'sample': len(chosen), 'seed': args.seed,
                          'dataset_alignment_errors_all_records': sum(bool(r.error) for r in records),
                          'alignment_failures': [{'sent_id': r.sent_id, 'error': r.error} for r in records if r.error]},
              'reader_budgets': {k: getattr(reader, k) for k in ('tag_beam_width', 'tag_max_candidates',
                               'parse_beam_width', 'parse_max_candidates', 'parse_ranking', 'max_expansions',
                               'max_sentence_expansions', 'max_alternatives', 'semantic_max_candidates',
                               'semantic_max_expansions', 'max_sentence_semantic_expansions')},
              'metrics': {**summarize(rows),
                          'inputs_with_no_acts': sum(r['proposals_with_acts'] == 0 for r in rows),
                          'inputs_with_retention_discards': sum(r['retention_discarded'] > 0 for r in rows),
                          'inputs_with_search_truncation': sum(r['search_truncated'] for r in rows)},
              'latency_ms': {'median': statistics.median(r['latency_ms'] for r in rows),
                             'total': sum(r['latency_ms'] for r in rows)},
              'limitations': ['Small smoke sample; not a generalization benchmark.',
                              'Oracle selects whole syntax candidates using gold annotations, not actual agent meaning selection.',
                              'First proposal is a retention order, not an endorsed interpretation.',
                              'Unaligned words/heads receive no full-denominator arc credit; unaffected arcs remain scoreable.',
                              'Conditional metrics exclude unrepresentable endpoints and must be read with their denominators.',
                              'Basic dependencies only; empty nodes and enhanced dependencies excluded.'],
              'sentences': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'sentences'}, indent=2))


if __name__ == '__main__':
    main()
