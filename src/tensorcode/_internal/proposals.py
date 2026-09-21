"""Owned text generation shared by cognitive tools; outputs remain proposals."""
from __future__ import annotations

import hashlib
import json

import torch


def proposal_prompt(inputs, task_key, *, template_version=1):
    """Serialize only task and identified evidence; supervision never enters memory."""
    if not isinstance(inputs, dict) or not isinstance(inputs.get(task_key), str) or not inputs[task_key].strip():
        raise ValueError(f'{task_key} must be nonempty text')
    if type(template_version) is not int or template_version not in (1, 2) or (template_version == 2 and task_key != 'question'):
        raise ValueError('unsupported proposal template version for task')
    evidence = inputs.get('evidence', [])
    if not isinstance(evidence, list):
        raise ValueError('evidence must be a list')
    ids = set()
    for item in evidence:
        if (not isinstance(item, dict) or not isinstance(item.get('source_id'), str)
                or not item['source_id'] or item['source_id'] in ids
                or not isinstance(item.get('text'), str) or not item['text'].strip()):
            raise ValueError('evidence requires unique source_id and nonempty text')
        ids.add(item['source_id'])
    instruction = ('Generate one declarative candidate explanation or answer to the question using the supplied evidence.'
                   if task_key == 'question' else 'Generate one proposed plan for the goal using the supplied evidence.')
    instruction += ' This is an uncertain proposal. Do not fabricate observations. Return only the candidate text.\n'
    if template_version == 2:
        instruction = ('Answer the question using only the supplied evidence. Write the answer as one complete sentence. '
                       'Do not include JSON or repeat the evidence.\n')
    return instruction + json.dumps({task_key: inputs[task_key], 'evidence': [
        {'source_id': item['source_id'], 'text': item['text']} for item in evidence]}, ensure_ascii=False)


def generate_proposals(generator, inputs, *, task_key, count=3, kind='hypothesis', max_count=16, template_version=1):
    if generator is None:
        raise ValueError('proposal generation capability is not configured; configure an owned generator')
    with generator._lock:
        return _generate_proposals(generator, inputs, task_key=task_key, count=count,
                                   kind=kind, max_count=max_count, template_version=template_version)


def _generate_proposals(generator, inputs, *, task_key, count, kind, max_count, template_version):
    if generator is None:
        raise ValueError('proposal generation capability is not configured; configure an owned generator')
    if type(max_count) is not int or max_count < 1 or type(count) is not int or not 1 <= count <= max_count:
        raise ValueError(f'count must be an integer between 1 and {max_count}')
    prompt = proposal_prompt(inputs, task_key, template_version=template_version)
    modes = [(module, module.training) for module in generator.modules()]
    try:
        generator.eval()
        with torch.no_grad():
            state = generator.encode_workspace([prompt])
            tokens = generator.decoder(state, context={
                'max_new_tokens': generator.config['max_new_tokens'],
                'do_sample': False, 'num_beams': count, 'num_return_sequences': count,
                'return_dict_in_generate': False})
            texts = generator.tokenizer.batch_decode(tokens, skip_special_tokens=True)
    finally:
        for module, training in modes:
            module.training = training
    if not isinstance(texts, list) or len(texts) > count or any(not isinstance(text, str) for text in texts):
        raise ValueError('generator returned malformed text sequences')
    identity = hashlib.sha256(json.dumps({'generator': generator.fingerprint,
                'proposal_template_version': template_version, 'task_key': task_key}, sort_keys=True).encode()).hexdigest()
    records, seen = [], set()
    for text in texts:
        text = text.strip()
        normalized = ' '.join(text.split()).casefold()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        records.append({'id': f'{kind}-' + hashlib.sha256((prompt + '\n' + text).encode()).hexdigest()[:20], 'text': text,
                        'origin': 'generated', 'generated_by': identity,
                        'proposal_template_version': template_version,
                        'generator_identity_kind': 'configuration_and_prompt_template_fingerprint',
                        'generator_configuration_fingerprint': generator.fingerprint,
                        'generator_foundation': generator.configuration().get('foundation'),
                        'epistemic_status': kind,
                        'source_ids': [item['source_id'] for item in inputs.get('evidence', [])],
                        'source_reference_kind': 'generation_context',
                        'input_truncated': len(generator.tokenizer(prompt)['input_ids']) > generator.config['max_input_tokens']})
    return records


def proposal_loss(generator, inputs, targets, *, task_key, template_version=1):
    if generator is None:
        raise ValueError('proposal generation capability is not configured; configure an owned generator')
    targets = [targets] if isinstance(targets, str) else targets
    if not isinstance(targets, (list, tuple)) or not targets or any(not isinstance(x, str) or not x.strip() for x in targets):
        raise ValueError('targets must be nonempty generation text')
    prompt = proposal_prompt(inputs, task_key, template_version=template_version)
    return generator.loss_batch([prompt] * len(targets), list(targets))
