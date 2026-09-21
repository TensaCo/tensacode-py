"""Replace one explicit verifier foundation, calibrate temperature, and compare QA.

Run bootstrap/evaluate on the authorized CUDA host. Historical receipt recovery
creates development inputs only. No classifier weights are trained here.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import random

FOUNDATION = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'
REVISION = '6f5cf0a2b59cabb106aca4c287eed12e357e90eb'
VERIFIER_PREFIX = 'investigator.verifier.'


def sibling(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def tensor_digests(module):
    import torch
    return {name: digest_json({'dtype': str(value.dtype), 'shape': list(value.shape),
                             'bytes': hashlib.sha256(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()})
            for name, value in module.state_dict().items()}


def assert_unchanged(before, after):
    old = {key: value for key, value in before.items() if not key.startswith(VERIFIER_PREFIX)}
    new = {key: value for key, value in after.items() if not key.startswith(VERIFIER_PREFIX)}
    if old != new:
        changed = sorted(key for key in old.keys() | new.keys() if old.get(key) != new.get(key))
        raise ValueError(f'non-verifier tensors changed: {changed}')


def without_verifier(config):
    result = copy.deepcopy(config)
    investigator = result['cognition']['investigator']
    for key in list(investigator):
        if key.startswith('verifier_'):
            del investigator[key]
    return result


def case_inputs(case):
    return {'question': case['question'], 'evidence': copy.deepcopy(case['evidence'])}


def recover_cases(report):
    """Gold targets remain scoring metadata, never supplied to the Chatbot."""
    cases = []
    for row in report['real_data']['records']:
        evidence = row['receipt']['cognition']['evidence']
        if not evidence:
            raise ValueError('historical record has no recoverable evidence')
        cases.append({'id': row['id'], 'question': row['question'], 'target': row['target'],
                      'evidence': [{key: item[key] for key in ('id', 'source_id', 'text')} for item in evidence],
                      'source_kind': 'historical_hotpotqa_oracle_support_development'})
    if not cases or len({case['id'] for case in cases}) != len(cases):
        raise ValueError('historical records must have unique nonempty IDs')
    return cases


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def execution_provenance(args):
    return {'script': Path(__file__).name, 'script_sha256': file_digest(__file__),
            'source_commit': args.source_commit,
            'helper_script_sha256': {name: file_digest(Path(__file__).with_name(name + '.py'))
                                    for name in ('train_verifier', 'evaluate_cognition')}}


def runtime():
    import torch
    if not torch.cuda.is_available():
        raise ValueError('bootstrap/evaluate requires the authorized CUDA host')
    torch.set_num_threads(8)
    torch.manual_seed(17)
    random.seed(17)


def make_verifier(foundation, repository, revision):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from tensorcode.tools.investigator import EvidenceVerifier
    if not Path(foundation).is_dir():
        raise ValueError('foundation must be a downloaded local directory')
    tokenizer = AutoTokenizer.from_pretrained(foundation, local_files_only=True, use_fast=True, trust_remote_code=False)
    if not tokenizer.is_fast:
        raise ValueError('foundation requires a serializable fast tokenizer')
    native = AutoModelForSequenceClassification.from_pretrained(
        foundation, local_files_only=True, use_safetensors=True, trust_remote_code=False)
    config = {'verifier_config': native.config.to_dict(),
              'verifier_tokenizer_json': tokenizer.backend_tokenizer.to_str(),
              'verifier_tokenizer_special_tokens': {k: str(v) for k, v in tokenizer.special_tokens_map.items() if isinstance(v, str)},
              'verifier_labels': sibling('train_verifier').label_mapping(native.config.id2label),
              'verifier_max_tokens': 192,
              'verifier_foundation': {'repository': repository, 'revision': revision}}
    # Native Transformers configs use integer id2label keys; owned artifacts use JSON.
    config = json.loads(json.dumps(config, allow_nan=False))
    verifier = EvidenceVerifier(config)
    verifier.model.load_state_dict(native.state_dict(), strict=True)
    return verifier


def bootstrap(args):
    import torch
    from tensorcode.tools.chatbot import Chatbot
    from safetensors.torch import save_file
    runtime()
    source = execution_provenance(args)
    train = sibling('train_verifier')
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    bot = Chatbot.from_pretrained(args.baseline, local_files_only=True).eval()
    before_config = bot.configuration()
    before = tensor_digests(bot)
    baseline_fingerprint = bot.fingerprint
    verifier = make_verifier(args.foundation, args.foundation_repo, args.revision)
    # Fresh buffers start at temperature=1, calibrated=False; old fit is discarded.
    if bool(verifier.calibration.calibrated) or float(verifier.calibration.temperature) != 1.:
        raise RuntimeError('new verifier unexpectedly carries calibration')
    bot.investigator.verifier = verifier
    for key in list(bot.investigator.config):
        if key.startswith('verifier_'):
            del bot.investigator.config[key]
    bot.investigator.config.update(verifier.configuration())
    if without_verifier(before_config) != without_verifier(bot.configuration()):
        raise RuntimeError('non-verifier configuration changed')
    records, calibration_source = train.load_records('validation', 256)
    foundation_weights = tensor_digests(verifier.model)
    verifier.to('cuda').eval()
    logits = train.predict(verifier, records, 8)
    labels = train.labels_tensor(verifier, records)
    fit = verifier.calibration.fit(logits, labels)
    verifier.to('cpu')
    if foundation_weights != tensor_digests(verifier.model):
        raise RuntimeError('temperature fitting changed foundation weights')
    after = tensor_digests(bot)
    assert_unchanged(before, after)
    bot.save_pretrained(output / 'model')
    restored = Chatbot.from_pretrained(output / 'model', local_files_only=True).eval()
    reloaded = tensor_digests(restored)
    if after != reloaded or bot.configuration() != restored.configuration():
        raise RuntimeError('complete artifact did not reload exactly')
    restored.investigator.verifier.to('cuda').eval()
    reload_logits = train.predict(restored.investigator.verifier, records[:8], 8)
    torch.testing.assert_close(reload_logits, logits[:8], rtol=0, atol=0)
    save_file({'calibration_logits': logits, 'calibration_labels': labels,
               'reload_logits': reload_logits}, str(output / 'calibration.safetensors'))
    manifest = {'execution_source': source, 'foundation': {'repository': args.foundation_repo, 'revision': args.revision,
                              'local_files': {str(p.relative_to(args.foundation)): file_digest(p) for p in sorted(args.foundation.rglob('*')) if p.is_file() and '.cache' not in p.parts}},
                'baseline': {'path': str(args.baseline), 'fingerprint': baseline_fingerprint},
                'model_fingerprint': bot.fingerprint,
                'settings': {'seed': 17, 'threads': 8, 'device': 'cuda', 'max_tokens': 192, 'batch_size': 8},
                'calibration': {'dataset': train.DATASET, 'revision': train.DATASET_REVISION,
                                'split': 'validation', 'count': 256, 'source': calibration_source, 'fit': fit,
                                'records': records, 'selection': 'first distinct labeled pairs in pinned parquet order'},
                'verification': {'non_verifier_config_equal': True, 'non_verifier_tensors_equal': True,
                                 'reload_all_tensors_equal': True, 'reload_config_equal': True, 'reload_logits_exact': True,
                                 'foundation_weights_unchanged_by_calibration': True},
                'component_tensor_digests': {'baseline': before, 'replacement': after, 'reload': reloaded},
                'limitations': ['Foundation capability is inherited; only temperature is fitted, no model weight training.',
                                'SNLI calibration is not evidence-QA calibration; foundation pretraining overlap is not independently certified.',
                                'This comparison does not establish workspace gains. No thresholds or source policies are changed.']}
    write_json(output / 'manifest.json', manifest)
    print(json.dumps({'model': str(output / 'model'), 'verification': manifest['verification']}), flush=True)


def evaluate(args):
    import torch
    from tensorcode.tools.chatbot import Chatbot
    runtime()
    source = execution_provenance(args)
    helper = sibling('evaluate_cognition')
    cases = helper.load_cases(args.cases)
    if len(cases) < 8:
        raise ValueError('fixed evaluation protocol requires at least eight cases')
    if args.output.exists() or args.output.with_suffix('.progress.jsonl').exists():
        raise FileExistsError('refusing to overwrite evaluation results')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    bot = Chatbot.from_pretrained(args.model, local_files_only=True).to('cuda').eval()
    with torch.no_grad():
        report = helper.evaluate(bot, cases, progress_path=args.output.with_suffix('.progress.jsonl'), control_count=8)
        report['authored_fixtures'] = helper.evaluate(bot, helper.authored_cases(), control_count=0)
        report['authored_fixtures']['expected_behavior'] = 'Explicit conflicting-source fixture; abstain rather than assert one incompatible alternative.'
    report.update({'execution_source': source, 'model_fingerprint': bot.fingerprint, 'model_path': str(args.model),
                   'cases_sha256': file_digest(args.cases), 'cases': cases,
                   'input_fingerprints': {case['id']: digest_json(case_inputs(case)) for case in cases},
                   'verifier_configuration': bot.investigator.verifier.configuration(),
                   'protocol': {'seed': 17, 'threads': 8, 'control_count': 8, 'control_selection': 'first eight cases in input order',
                                'role': args.role, 'primary_count': len(cases)},
                   'manual_factual_review': {'status': 'pending', 'instructions': 'Review all non-abstained primary answers against full evidence and target; lexical containment and NLI approval are not accuracy.'},
                   'limitations': ['Oracle supporting passages; no learned retrieval claim.',
                                   'Inherited verifier comparison; no workspace advantage established.',
                                   'Authored source-withdrawal controls are not genuine contradictory public evidence.']})
    manifest = args.cases.with_name('data-manifest.json')
    if manifest.exists():
        report['data_manifest'] = json.loads(manifest.read_text())
    write_json(args.output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    build = sub.add_parser('bootstrap')
    build.add_argument('--baseline', type=Path, required=True)
    build.add_argument('--foundation', type=Path, required=True)
    build.add_argument('--foundation-repo', default=FOUNDATION)
    build.add_argument('--revision', default=REVISION)
    build.add_argument('--output', type=Path, required=True)
    build.add_argument('--source-commit', help='Explicit source Git commit; script hashes also capture uncommitted changes')
    run = sub.add_parser('evaluate')
    run.add_argument('--model', type=Path, required=True)
    run.add_argument('--source-commit', help='Explicit source Git commit; script hashes also capture uncommitted changes')
    run.add_argument('--cases', type=Path, required=True)
    run.add_argument('--output', type=Path, required=True)
    run.add_argument('--role', choices=['development', 'final'], required=True)
    recover = sub.add_parser('recover-development')
    recover.add_argument('--report', type=Path, required=True)
    recover.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'bootstrap':
        bootstrap(args)
    elif args.command == 'evaluate':
        evaluate(args)
    else:
        cases = recover_cases(json.loads(args.report.read_text()))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as stream:
            stream.writelines(json.dumps(case) + '\n' for case in cases)
        write_json(args.output.with_suffix('.provenance.json'), {'source_report': str(args.report),
                   'source_report_sha256': file_digest(args.report), 'role': 'development only; previously inspected historical cases',
                   'cases_sha256': file_digest(args.output), 'count': len(cases)})


if __name__ == '__main__':
    main()
