"""Evaluate an owned cognitive Chatbot on explicit evidence, with honest controls.

Real HotpotQA rows use oracle supporting passages. Answer coverage and response
fidelity below are lexical diagnostics, not semantic correctness judgments.
Authored contradictions are reported separately from public-data measurements.
Run substantial models on the designated evaluation host, not during unit tests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import string


def normalize(text):
    text = text.casefold().translate(str.maketrans('', '', string.punctuation))
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', text).split())


def prepare_hotpot(row):
    support = set(row['supporting_facts']['title'])
    evidence = [{'id': f'doc-{i}', 'source_id': title, 'text': title + '\n' + ' '.join(sentences)}
                for i, (title, sentences) in enumerate(zip(row['context']['title'], row['context']['sentences']))
                if title in support]
    if not evidence:
        raise ValueError('no oracle supporting evidence')
    return {'id': row['id'], 'question': row['question'], 'target': row['answer'], 'evidence': evidence,
            'source_kind': 'public_hotpotqa_oracle_support'}


def load_cases(path):
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    ids = [r['id'] for r in records]
    if not records or len(ids) != len(set(ids)):
        raise ValueError('evaluation requires unique nonempty records')
    return records


def summarize(records):
    n = len(records)
    if not n:
        return {'count': 0}
    return {'count': n, 'failed_calls': sum('error' in r for r in records), 'answer_exact_match': sum(r['answer_exact_match'] for r in records) / n,
            'candidate_answer_substring_coverage': sum(r['candidate_answer_substring_coverage'] for r in records) / n,
            'abstention_rate': sum(r['abstained'] for r in records) / n,
            'default_abstention_text_rate': sum(r.get('default_abstention_text', False) for r in records) / n,
            'selected_text_preserved_rate': sum(r['selected_text_preserved'] for r in records) / n,
            'selected_contradiction_veto_violations': sum(r['contradicted_selection'] for r in records),
            'generation_truncations': sum(r['input_truncated'] for r in records),
            'realization_nli_support_mean_on_selected': (sum(r['realization_nli'][0]['distribution']['support'] for r in records if r.get('realization_nli')) / sum(bool(r.get('realization_nli')) for r in records)) if any(r.get('realization_nli') for r in records) else None,
            'uncalibrated_verifier_records': sum(r.get('verifier_calibrated') is False for r in records)}


def measure(case, answer, receipt):
    cognition = receipt['cognition']
    candidates = cognition['candidates']
    selected_id = cognition.get('selected_id')
    selected = next((c for c in candidates if c['id'] == selected_id), None)
    # Full selected declaration must appear in realization; abstentions do not
    # count as faithful realizations. This intentionally undercounts paraphrases.
    target = normalize(case['target'])
    return {'id': case['id'], 'question': case['question'], 'target': case['target'],
            'answer': answer, 'answer_exact_match': normalize(answer) == target,
            'candidate_answer_substring_coverage': bool(target) and any((' ' + target + ' ') in (' ' + normalize(c['text']) + ' ') for c in candidates),
            'selected_text_preserved': bool(selected) and normalize(selected['text']) in normalize(answer),
            'abstained': cognition['abstained'],
            'default_abstention_text': answer == 'I do not have enough supported evidence to answer.',
            'contradicted_selection': bool(selected) and any(v['distribution']['contradiction'] > cognition['policy']['max_contradiction'] for v in selected['verifications']),
            'input_truncated': receipt.get('input_truncated', False) or any(c.get('input_truncated', False) for c in candidates),
            'verifier_calibrated': all(v.get('calibrated', False) for c in candidates for v in c.get('verifications', [])) if any(c.get('verifications') for c in candidates) else None,
            'receipt': receipt}


def evaluate(bot, cases, *, progress_path=None):
    records, controls = [], []
    def call(session, value, case):
        try:
            answer = session(value)
            result = measure(case, answer, session.last_result)
            cognition = session.last_result['cognition']
            selected = next((item for item in cognition['candidates'] if item['id'] == cognition.get('selected_id')), None)
            if selected is not None:
                result['realization_nli'] = bot.investigator.verifier.verify(answer, [{'source_id': selected['id'], 'text': selected['text']}])
            return result
        except (ValueError, RuntimeError) as error:
            return {'id': case['id'], 'question': case['question'], 'target': case['target'], 'answer': None,
                    'error': {'type': type(error).__name__, 'message': str(error)},
                    'answer_exact_match': False, 'candidate_answer_substring_coverage': False,
                    'selected_text_preserved': False, 'abstained': False, 'contradicted_selection': False,
                    'input_truncated': False}
    for case in cases:
        session = bot.new_session()
        value = {'question': case['question'], 'evidence': case['evidence']}
        records.append(call(session, value, case))
        # Fresh-session source omission isolates evidence availability without
        # claiming to implement historical source deletion.
        empty = bot.new_session()
        omission = call(empty, {'question': case['question']}, case)
        replacement = {'text': 'This source is unavailable and supplies no evidence about the question.'}
        revised = [dict(replacement, evidence_id=item['id']) for item in case['evidence']] if case['evidence'] else []
        replacement_result = call(session, {'question': case['question'], 'revisions': revised}, case)
        controls.append({'id': case['id'], 'omission': omission,
                         'replacement': replacement_result,
                         'replacement_kind': 'authored source-withdrawal notice, not real-world evidence',
                         'state_revision_before': records[-1].get('receipt', {}).get('cognition', {}).get('state_revision'),
                         'state_revision_after': replacement_result.get('receipt', {}).get('cognition', {}).get('state_revision')})
        if progress_path is not None:
            with Path(progress_path).open('a') as stream:
                stream.write(json.dumps({'record': records[-1], 'controls': controls[-1]}) + '\n')
        print(json.dumps({'id': case['id'], 'answer': records[-1]['answer'], 'abstained': records[-1]['abstained'], 'error': records[-1].get('error')}), flush=True)
    return {'real_data': {'metrics': summarize(records), 'records': records},
            'controls': {'records': controls, 'omission': summarize([r['omission'] for r in controls]),
                         'replacement': summarize([r['replacement'] for r in controls])}}


def evaluate_memory(bot, cases):
    """Retrieval over oracle passages from these diagnostic questions only."""
    from tensorcode.runtime.cognition import LearnedEpisodicMemory
    from tensorcode.runtime.cognitive_state import Evidence
    corpus = [(case, item, case['id'] + ':' + item['id']) for case in cases for item in case['evidence']]
    memory = LearnedEpisodicMemory(bot.investigator, capacity=max(1, len(corpus)))
    for case, item, identity in corpus:
        memory.remember(Evidence(identity, item['text'], item['source_id']), episode_id=case['id'])
    records = []
    for case in cases:
        relevant = {case['id'] + ':' + item['id'] for item in case['evidence']}
        learned = [hit.evidence.id for hit in memory.retrieve(case['question'], k=5)]
        tokens = set(re.findall(r'\w+', case['question'].casefold()))
        lexical = [identity for _, _, identity in sorted(corpus, key=lambda row:
                   -len(tokens & set(re.findall(r'\w+', row[1]['text'].casefold()))))[:5]]
        records.append({'id': case['id'], 'relevant_ids': sorted(relevant), 'learned_ids': learned, 'lexical_ids': lexical,
                        'learned_hit_at_1': bool(learned and learned[0] in relevant),
                        'learned_hit_at_5': bool(set(learned) & relevant),
                        'lexical_hit_at_1': bool(lexical and lexical[0] in relevant),
                        'lexical_hit_at_5': bool(set(lexical) & relevant)})
    metrics = {key: sum(record[key] for record in records) / len(records) for key in
               ['learned_hit_at_1', 'learned_hit_at_5', 'lexical_hit_at_1', 'lexical_hit_at_5']}
    return {'metrics': metrics, 'records': records, 'corpus_size': len(corpus),
            'limitations': 'Oracle supporting-passage corpus built from these diagnostic questions, not open-corpus retrieval. Ranker encoder was trained on a separate HotpotQA subset; no fitting here.'}


def authored_cases():
    return [{'id': 'authored-conflict', 'question': 'Is the door open?', 'target': 'insufficient evidence',
             'evidence': [{'id': 'report-1', 'source_id': 'authored-observer-A', 'text': 'The door is open.'},
                          {'id': 'report-2', 'source_id': 'authored-observer-B', 'text': 'The door is closed and is not open.'}],
             'source_kind': 'authored_mechanism_fixture'}]


def assemble(ranker_path, language_repo, language_revision, verifier_directory, output, *, generator_path=None):
    """Own and serialize every component, retaining fitted verifier calibration."""
    import torch
    from safetensors.torch import load_file
    from tensorcode.tools.chatbot import Chatbot
    from tensorcode.tools.investigator import Investigator
    torch.manual_seed(17)
    ranker = Investigator.from_pretrained(ranker_path)
    language = (Chatbot.from_pretrained(language_repo) if Path(language_repo).is_dir() and (Path(language_repo) / 'tensorcode_config.json').exists()
                else Chatbot.from_foundation(language_repo, revision=language_revision, local_files_only=True,
                                            max_input_tokens=1024, max_new_tokens=96))
    generator = Chatbot.from_pretrained(generator_path) if generator_path else language
    verifier_directory = Path(verifier_directory)
    verifier_config = json.loads((verifier_directory / 'verifier_config.json').read_text())
    config = dict(ranker.configuration(), generator=generator.configuration(), **verifier_config)
    investigator = Investigator(config)
    investigator.rank.load_state_dict(ranker.rank.state_dict())
    investigator.generator.load_state_dict(generator.state_dict())
    investigator.verifier.load_state_dict(load_file(str(verifier_directory / 'verifier.safetensors')))
    config = dict(language.configuration(), cognition={'investigator': investigator.configuration(), 'proposal_count': 3,
                  'policy': {'min_support': .7, 'max_contradiction': .2, 'max_unknown': .3}})
    model = Chatbot(config)
    missing, unexpected = model.load_state_dict(language.state_dict(), strict=False)
    if unexpected or any(not key.startswith('investigator.') for key in missing):
        raise ValueError('language component transfer mismatch')
    model.investigator.load_state_dict(investigator.state_dict())
    model.save_pretrained(output)
    provenance = {'initialization_seed': 17, 'ranker': str(ranker_path), 'language': str(language_repo), 'language_revision': language_revision, 'generator': str(generator_path or language_repo),
                  'verifier_directory': str(verifier_directory), 'verifier_calibrated': bool(model.investigator.verifier.calibration.calibrated),
                  'policy_origin': 'authored fixed thresholds; no evaluation tuning',
                  'component_weight_sha256': {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in
                      [('ranker', Path(ranker_path) / 'model.safetensors'), ('language', Path(language_repo) / 'model.safetensors'), ('generator', Path(generator_path or language_repo) / 'model.safetensors'), ('verifier', verifier_directory / 'verifier.safetensors')] if path.exists()}}
    (Path(output) / 'assembly-provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--role', choices=['diagnostic', 'final'], default='diagnostic')
    parser.add_argument('--ranker')
    parser.add_argument('--language-repo', default='google/flan-t5-base')
    parser.add_argument('--language-revision')
    parser.add_argument('--generator-model', type=Path)
    parser.add_argument('--verifier-directory', type=Path)
    args = parser.parse_args()
    import torch
    from tensorcode.tools.chatbot import Chatbot
    torch.set_num_threads(8)
    if args.ranker:
        if not args.verifier_directory or (not args.language_revision and not Path(args.language_repo).is_dir()):
            parser.error('assembly requires verifier directory and pinned language revision')
        bot = assemble(args.ranker, args.language_repo, args.language_revision, args.verifier_directory, args.model, generator_path=args.generator_model)
    else:
        bot = Chatbot.from_pretrained(args.model)
    bot.to(args.device).eval()
    cases = load_cases(args.cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    progress = args.output.with_suffix('.progress.jsonl')
    progress.write_text('')
    with torch.no_grad():
        report = evaluate(bot, cases, progress_path=progress)
        report['authored_fixtures'] = evaluate(bot, authored_cases())
        report['authored_fixtures']['conflict_cases_with_selected_candidate'] = sum(not row['abstained'] and 'error' not in row for row in report['authored_fixtures']['real_data']['records'])
        report['authored_fixtures']['expected_behavior'] = 'Authored mutually conflicting sources should prevent an unqualified selection; this is a fixture, not a public benchmark.'
        report['episodic_retrieval'] = evaluate_memory(bot, cases)
    manifest = args.cases.with_name('data-manifest.json')
    if manifest.exists():
        report['data_manifest'] = json.loads(manifest.read_text())
    report['model_fingerprint'] = bot.fingerprint
    report['foundation'] = bot.configuration().get('foundation')
    report['evaluation_role'] = ('diagnostic development: observed results may motivate later model changes; not a final untouched benchmark'
                                 if args.role == 'diagnostic' else 'fixed-configuration final evaluation; do not tune on these results')
    report['cases_sha256'] = hashlib.sha256(args.cases.read_bytes()).hexdigest()
    report['limitations'] = ['Oracle supporting passages, not learned retrieval.',
        'Lexical answer coverage and fidelity diagnostics do not establish semantic correctness; realization NLI uses the same separately calibrated SNLI verifier and is an additional model judgment.',
        'NLI calibration is inherited from a separate SNLI calibration split, not evidence-QA calibration.',
        'Authored threshold policy screens model scores; acceptance is not proof of truth.',
        'Foundation generation workspace is unadapted unless separately documented.']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
