"""Historical-case foundation/workspace diagnostic; never a production policy.

Only the explicitly supplied cases file is read. No final-set discovery, model
training, model publication, or model download occurs. Reference metadata is
recorded for subsequent review and never included in generation prompts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def prompts(case, *, proposal_style="existing"):
    from tensorcode._internal.proposals import proposal_prompt
    # Historical evaluation cases use unique evidence IDs and document source
    # IDs separately. The existing proposal protocol receives the unique IDs.
    value = {'question': case['question'], 'evidence': [
        {'source_id': row.get('id', row.get('source_id')), 'text': row['text']}
        for row in case['evidence']]}
    proposed = proposal_prompt(value, 'question')  # Also validates inputs.
    if proposal_style == 'evidence_qa':
        proposed = ('Answer the question using only the supplied evidence. Write the answer as one complete sentence. '
                    'Do not include JSON or repeat the evidence.\n' + json.dumps(value, ensure_ascii=False))
    elif proposal_style != 'existing':
        raise ValueError('unknown proposal style')
    direct = ('Answer the question using only the supplied evidence. Return a concise answer.\n'
              + json.dumps(value, ensure_ascii=False))
    return direct, proposed


def load_cases(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    ids = [row.get('id') for row in rows]
    if not rows or any(not isinstance(key, str) or not key.strip() for key in ids) or len(ids) != len(set(ids)):
        raise ValueError('cases require unique nonempty IDs and at least one case')
    for row in rows:
        prompts(row)
    return rows


def proposal_batch(model, prompt, *, workspace_ablation):
    """Existing native proposal decoding protocol, preserving all three beams."""
    import torch
    with torch.no_grad():
        state = model.encode_workspace([prompt], workspace_ablation=workspace_ablation)
        tokens = model.decoder(state, context={
            'max_new_tokens': model.config['max_new_tokens'], 'do_sample': False,
            'num_beams': 3, 'num_return_sequences': 3, 'return_dict_in_generate': False})
        texts = model.tokenizer.batch_decode(tokens, skip_special_tokens=True)
    if not isinstance(texts, list) or len(texts) != 3 or any(not isinstance(text, str) for text in texts):
        raise ValueError('native three-beam decoding must return three text sequences')
    return texts


def artifact_hashes(directory):
    return {str(path.relative_to(directory)): sha256(path)
            for path in sorted(directory.rglob('*')) if path.is_file()
            and path.suffix in {'.json', '.safetensors', '.model', '.txt'}}


def probe_case(model, case, *, proposal_style="existing"):
    """Yield each completed mode separately so progress survives interruption."""
    import torch
    direct, proposed = prompts(case, proposal_style=proposal_style)
    modes = [('direct_answer_bypass', direct, 'bypass', 1),
             ('declarative_proposals_bypass', proposed, 'bypass', 3),
             ('declarative_proposals_active', proposed, None, 3)]
    for mode, prompt, ablation, beams in modes:
        count = len(model.tokenizer(prompt, truncation=False)['input_ids'])
        record = {'case_id': case['id'], 'mode': mode, 'prompt': prompt,
                  'source_ids': [row.get('id', row.get('source_id')) for row in case['evidence']],
                  'input_token_count': count, 'max_input_tokens': model.config['max_input_tokens'],
                  'input_truncated': count > model.config['max_input_tokens'],
                  'max_new_tokens': model.config['max_new_tokens'],
                  'workspace_ablation': ablation, 'do_sample': False,
                  'num_beams': beams, 'num_return_sequences': beams}
        torch.cuda.synchronize()
        started = time.perf_counter()
        try:
            if beams == 1:
                texts = model.generate_batch([prompt], workspace_ablation='bypass')
                if not isinstance(texts, list) or len(texts) != 1 or not isinstance(texts[0], str):
                    raise ValueError('direct generation must return one text')
            else:
                texts = proposal_batch(model, prompt, workspace_ablation=ablation)
            torch.cuda.synchronize()
            record['candidates'] = [{'beam_index': index, 'text': text,
                'retokenized_output_token_count': len(model.tokenizer(text, add_special_tokens=False,
                                                                     truncation=False)['input_ids'])}
                for index, text in enumerate(texts)]
            record['output_count_semantics'] = 'decoded text retokenized without special tokens; not original generated sequence length'
        except Exception as exc:
            record['error'] = {'type': type(exc).__name__, 'message': str(exc)}
            torch.cuda.empty_cache()
        record['elapsed_seconds'] = time.perf_counter() - started
        yield record


def run(args):
    # Make accidental remote fallback impossible in either owned loading route.
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import platform
    import torch
    from tensorcode.tools.chatbot import Chatbot
    if not torch.cuda.is_available():
        raise RuntimeError('foundation-scale diagnostics require the authorized CUDA host')
    if args.foundation and (not args.revision or args.component != 'language'):
        raise ValueError('--foundation requires --revision and component language')
    if args.model and (args.revision or args.foundation_repository):
        raise ValueError('owned models use artifact provenance; foundation options require --foundation')
    directory = Path(args.foundation or args.model).expanduser().resolve()
    if not directory.is_dir():
        raise ValueError('model/foundation must be an existing local directory')
    if not any(directory.glob('*.safetensors')):
        raise ValueError('local model/foundation must contain safetensors weights')
    cases = load_cases(args.cases)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    random.seed(20260921)
    torch.manual_seed(20260921)
    torch.cuda.manual_seed_all(20260921)
    torch.set_num_threads(8)
    report = {'role': 'historical development diagnostic; not production policy or held-out final evidence',
              'case_file': str(Path(args.cases).resolve()), 'case_file_sha256': sha256(args.cases),
              'script_sha256': sha256(__file__), 'case_count': len(cases),
              'case_ids': [row['id'] for row in cases], 'seed': 20260921,
              'host': platform.node(), 'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
              'component': args.component, 'proposal_style': args.proposal_style, 'artifact_path': str(directory),
              'artifact_files_sha256': artifact_hashes(directory),
              'requested_dtype': 'torch.bfloat16', 'frozen': True,
              'limitations': ['historical known cases only; no fresh final evaluation',
                              'foundation pretraining exposure unknown',
                              'raw generation is not source verification or a production response',
                              'workspace ablation does not alter the current cognition objective'],
              'records': []}
    write_json(output / 'manifest.json', report)
    if args.foundation:
        owner = Chatbot.from_foundation(directory, revision=args.revision, local_files_only=True)
        model = owner
        report['foundation_request'] = {'repository': args.foundation_repository or str(directory),
                                       'revision': args.revision, 'local_path': str(directory),
                                       'provenance_kind': 'caller-supplied repository/revision; local asset hashes recorded'}
        report['workspace_status'] = 'new random workspace and memory projection; untrained'
    else:
        owner = Chatbot.from_pretrained(directory, local_files_only=True)
        model = owner if args.component == 'language' else getattr(owner.investigator, 'generator', None)
        if model is None:
            raise ValueError('owned Chatbot has no configured investigator generator')
        report['workspace_status'] = 'loaded owned artifact; training competence not inferred from loading'
    model.to(device='cuda', dtype=torch.bfloat16).eval().requires_grad_(False)
    report['parameter_dtypes'] = sorted({str(value.dtype) for value in model.parameters()})
    report['foundation'] = model.configuration().get('foundation')
    report['configuration'] = model.configuration()
    # generate_batch leaves beam count to the native GenerationConfig. Fix the
    # direct-answer route to one deterministic sequence and record this override.
    model.foundation.generation_config.num_beams = 1
    model.foundation.generation_config.num_return_sequences = 1
    report['generation_overrides'] = {'num_beams': 1, 'num_return_sequences': 1,
        'proposals_explicit_context': {'num_beams': 3, 'num_return_sequences': 3, 'do_sample': False}}
    write_json(output / 'manifest.json', report)
    with (output / 'progress.jsonl').open('x', encoding='utf-8') as progress:
        for case in cases:
            records = []
            for record in probe_case(model, case, proposal_style=args.proposal_style):
                progress.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
                progress.flush()
                records.append(record)
            report['records'].append({'id': case['id'],
                'reference_metadata': {key: value for key, value in case.items() if key not in ('question', 'evidence')},
                'outputs': records})
            print(json.dumps({'case_id': case['id'], 'completed_cases': len(report['records']),
                              'total_cases': len(cases), 'errors': sum('error' in row for row in records)}), flush=True)
    report['elapsed_seconds'] = time.perf_counter() - started
    report['error_count'] = sum('error' in mode for row in report['records'] for mode in row['outputs'])
    write_json(output / 'report.json', report)
    if report['error_count']:
        raise RuntimeError('diagnostic completed with generation errors; inspect report.json')
    return report


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    source = result.add_mutually_exclusive_group(required=True)
    source.add_argument('--foundation', help='existing local foundation snapshot')
    source.add_argument('--model', help='existing local owned Chatbot artifact')
    result.add_argument('--component', choices=('language', 'investigator-generator'), default='language')
    result.add_argument('--revision', help='required provenance revision for --foundation')
    result.add_argument('--foundation-repository', help='explicit foundation repository provenance')
    result.add_argument('--proposal-style', choices=('existing', 'evidence_qa'), default='existing')
    result.add_argument('--cases', required=True, help='historical known-case JSONL; every row is used')
    result.add_argument('--output', required=True, help='new diagnostic directory')
    return result


if __name__ == '__main__':
    run(parser().parse_args())
