"""Complete-tool development diagnostic with explicit owned component replacement."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path


def configure(bot, generator, *, scope, max_tokens, template_version):
    if bot.investigator is None or bot.investigator.verifier is None:
        raise ValueError('requires a complete cognitive chatbot')
    if scope not in ('source', 'joint') or type(max_tokens) is not int or max_tokens < 1:
        raise ValueError('invalid verification configuration')
    if type(template_version) is not int or template_version not in (1, 2):
        raise ValueError('invalid template version')
    if generator is not None:
        if generator.investigator is not None:
            raise ValueError('proposal generator cannot recursively own cognition')
        bot.investigator.generator = generator
    bot.investigator.config.update(verification_scope=scope, verifier_max_tokens=max_tokens,
                                  proposal_template_version=template_version)
    bot.investigator.verifier.config['verifier_max_tokens'] = max_tokens
    bot.investigator.verifier.max_tokens = max_tokens
    # Ranking records the complete investigator configuration in trace identity.
    bot.investigator.rank.config = bot.investigator.configuration()
    bot.reset_session()
    return bot


def run(args):
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import torch
    from tensorcode.tools.chatbot import Chatbot
    if not torch.cuda.is_available():
        raise RuntimeError('real-model diagnostic requires authorized CUDA host')
    if args.foundation and not args.revision:
        raise ValueError('foundation revision is required')
    if not Path(args.model).is_dir() or (args.foundation and not Path(args.foundation).is_dir()):
        raise ValueError('models must be explicit existing local directories')
    torch.set_num_threads(8)
    torch.manual_seed(20260921)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    module_path = Path(__file__).resolve().parents[2] / 'examples/evaluate_cognition.py'
    spec = importlib.util.spec_from_file_location('cognitive_evaluation', module_path)
    evaluation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluation)
    cases = evaluation.load_cases(args.cases)
    bot = Chatbot.from_pretrained(args.model, local_files_only=True)
    original_config = bot.configuration()
    generator = (Chatbot.from_foundation(args.foundation, revision=args.revision,
                 local_files_only=True, max_input_tokens=512, max_new_tokens=64)
                 if args.foundation else None)
    configure(bot, generator, scope=args.scope, max_tokens=args.max_tokens,
              template_version=args.template_version)
    # Preserve the original ranker, verifier and realizer dtypes. Only the new
    # generator uses bf16, matching the foundation-scale diagnostic.
    if generator is not None:
        generator.to(dtype=torch.bfloat16)
    bot.eval().requires_grad_(False)
    bot.save_pretrained(output / 'model')
    manifest = {'role':'historical development; no final cases or checkpoint promotion',
                'case_sha256': hashlib.sha256(Path(args.cases).read_bytes()).hexdigest(),
                'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'configuration':bot.configuration(), 'original_configuration':original_config,
                'base_model':str(Path(args.model).resolve()),
                'foundation': args.foundation, 'revision':args.revision,
                'generator_dtype':str(next(bot.investigator.generator.parameters()).dtype),
                'seed':20260921, 'control_count':args.control_count,
                'limitations':['Inherited generator capability; new workspace untrained.',
                               'NLI approval and lexical metrics are not correctness labels.',
                               'Existing source-pair calibration transferred to joint inputs, not validated joint calibration.']}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    bot.to('cuda')
    with torch.no_grad():
        report = evaluation.evaluate(bot, cases, progress_path=output/'progress.jsonl',
                                     control_count=args.control_count)
        report['authored_fixtures'] = evaluation.evaluate(bot, evaluation.authored_cases())
        report['episodic_retrieval'] = evaluation.evaluate_memory(bot, cases)
    report['manifest'] = manifest
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['real_data']['metrics']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--foundation')
    parser.add_argument('--revision')
    parser.add_argument('--cases', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--scope', choices=('source','joint'), default='joint')
    parser.add_argument('--max-tokens', type=int, default=512)
    parser.add_argument('--template-version', type=int, choices=(1,2), default=2)
    parser.add_argument('--control-count', type=int, default=8)
    run(parser.parse_args())
