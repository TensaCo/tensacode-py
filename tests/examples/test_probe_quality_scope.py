"""Tiny owned native fixtures verify experiment mechanics, not judgment quality."""
import importlib.util
import json
from pathlib import Path
import runpy

import pytest
import torch


def module():
    path = Path(__file__).parents[2] / '.development/experiments/probe_quality_scope.py'
    spec = importlib.util.spec_from_file_location('quality_scope_test', path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


def tiny():
    from tensorcode.tools.chatbot import Chatbot
    config = runpy.run_path(str(Path(__file__).parents[1] / 'models/test_chatbot_model.py'))['tiny_config']()
    config['max_input_tokens'] = 512
    tokenizer = json.loads(config['tokenizer_json']); tokenizer['model']['vocab'].update(yes=8, no=9)
    config['tokenizer_json'] = json.dumps(tokenizer); config['foundation_config']['vocab_size'] = 10
    return Chatbot(config).eval()


ROW = {'question': 'hello?', 'candidate': 'world', 'evidence': [{'id': 's', 'source_id': 's', 'text': 'hello world'}],
       'targets': {'support': True}, 'reference': 'GOLD-SECRET', 'review': 'REVIEW-SECRET'}


def test_prompts_are_fixed_and_do_not_leak_annotations():
    probe = module()
    assert probe.INSTRUCTIONS['completeness'] == probe.probe.INSTRUCTIONS['completeness']
    for name in ('original', 'full_proposition'):
        prompts = probe.prompts(ROW, prompt_set=name)
        assert set(prompts) == {'support', 'completeness', 'constraints'}
        for text in prompts.values():
            value = json.loads(text.split('\n', 1)[1])
            assert set(value) == {'question', 'candidate', 'evidence'}
            assert 'GOLD-SECRET' not in text and 'REVIEW-SECRET' not in text and 'targets' not in text
    with pytest.raises(ValueError):
        probe.prompts(ROW, prompt_set='adapted_after_seeing_labels')


def test_original_scores_exactly_match_shared_probe_without_weight_changes():
    probe = module(); model = tiny()
    before = probe.probe.state_digest(model.state_dict())
    inputs = probe.helper.model_inputs(ROW)
    for mode in (None, 'bypass'):
        expected = probe.probe.assess(model, inputs, yes_id=8, no_id=9, workspace_ablation=mode)
        actual = probe.assess(model, ROW, prompt_set='original', yes_id=8, no_id=9, workspace_ablation=mode)
        assert actual == expected
        probe.assess(model, ROW, prompt_set='full_proposition', yes_id=8, no_id=9, workspace_ablation=mode)
    assert probe.probe.state_digest(model.state_dict()) == before
    assert probe.label_ids(model) == {'yes': [8], 'no': [9]}


def test_overflow_has_no_scores_or_model_call(monkeypatch):
    probe = module(); model = tiny(); model.config['max_input_tokens'] = 2
    monkeypatch.setattr(model, 'encode_workspace', lambda *a, **k: pytest.fail('overflow called model'))
    result = probe.assess(model, ROW, prompt_set='full_proposition', yes_id=8, no_id=9)
    assert result['input_truncated'] and result['scores'] is None


def test_archived_baseline_comparison_rejects_changed_receipts():
    probe = module()
    receipt = {'id': 'r', 'input_truncated': False, 'input_token_counts': dict.fromkeys(probe.INSTRUCTIONS, 2),
               'scores': dict.fromkeys(probe.INSTRUCTIONS, .5), 'bypass_scores': dict.fromkeys(probe.INSTRUCTIONS, .6)}
    archived = {'splits': {split: {'records': [receipt], 'excluded': []} for split in ('calibration', 'development')}}
    probe.compare_archived(archived, archived['splits'])
    altered = json.loads(json.dumps(archived['splits'])); altered['development']['records'][0]['scores']['support'] = .4
    with pytest.raises(ValueError, match='baseline'):
        probe.compare_archived(archived, altered)


def test_common_subset_coverage_excludes_new_overflow():
    probe = module()
    rows = [{'id': key, 'targets': dict.fromkeys(probe.INSTRUCTIONS, True)} for key in ('a', 'b')]
    scores = dict.fromkeys(probe.INSTRUCTIONS, .8)
    first = {'excluded': [], 'records': [{'id': row['id'], 'input_truncated': False, 'scores': scores, 'bypass_scores': scores} for row in rows]}
    second = json.loads(json.dumps(first))
    second['excluded'] = ['b']; second['records'][1].update(input_truncated=True, scores=None, bypass_scores=None)
    comparison = probe.comparison(rows, first, second)
    assert comparison['common_eligible_ids'] == ['a']
    assert comparison['newly_truncated_ids'] == ['b']
    for condition in comparison['common_eligible_gates'].values():
        for result in condition.values():
            assert result['coverage'] == {'total': 1, 'scored': 1, 'excluded': 0, 'fraction': 1.0}


def test_control_receipts_preserve_review_and_unknown_labels_outside_prompts():
    probe = module()
    row = dict(ROW, id='variant', original_id='anchor', intervention='evidence-free',
               targets={'support': False, 'completeness': None, 'constraints': None},
               review={'caveat': 'review-only caveat'})
    scored = {'id': row['id'], 'input_truncated': False,
              'input_token_counts': dict.fromkeys(probe.INSTRUCTIONS, 2),
              'scores': dict.fromkeys(probe.INSTRUCTIONS, .2), 'bypass_scores': dict.fromkeys(probe.INSTRUCTIONS, .3)}
    result = probe.controls_records([row], {'records': [scored]})[0]
    assert result['role'] == 'variant' and result['intervention'] == 'evidence-free'
    assert result['review'] == row['review'] and result['targets'] == row['targets']
    assert all('review-only caveat' not in prompt for prompt in probe.prompts(row, prompt_set='full_proposition').values())
    with pytest.raises(ValueError, match='label IDs'):
        probe.assess(tiny(), ROW, prompt_set='original', yes_id=9, no_id=8)
