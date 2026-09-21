import importlib.util
from pathlib import Path

import pytest
import torch


def runner():
    path = Path(__file__).parents[2] / '.development/experiments/probe_foundation_scale.py'
    spec = importlib.util.spec_from_file_location('probe_foundation_scale', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def case():
    return {'id': 'known-1', 'question': 'Which place?',
            'evidence': [{'id': 'e1', 'source_id': 'document', 'text': 'Observed text.'}],
            'target': 'SECRET_GOLD', 'reference_answer': 'SECRET_REFERENCE',
            'targets': {'support': True}, 'rationale': 'SECRET_REVIEW'}


def test_prompts_use_only_question_and_evidence():
    mod = runner()
    direct, proposals = mod.prompts(case())
    assert 'Which place?' in direct and 'Observed text.' in direct
    assert 'Which place?' in proposals and 'Observed text.' in proposals
    assert all(secret not in text for text in (direct, proposals)
               for secret in ('SECRET', 'targets', 'support', 'rationale', 'reference_answer'))
    from tensorcode._internal.proposals import proposal_prompt
    assert proposals == proposal_prompt({'question': 'Which place?', 'evidence': [
        {'source_id': 'e1', 'text': 'Observed text.'}]}, 'question')


def test_native_proposal_protocol_keeps_every_beam_and_ablation():
    mod = runner()
    calls = []
    class Tokenizer:
        def batch_decode(self, tokens, **kwargs):
            assert kwargs == {'skip_special_tokens': True}
            return ['repeat', 'repeat', '']
    class Model:
        config = {'max_new_tokens': 17}
        tokenizer = Tokenizer()
        def encode_workspace(self, inputs, *, workspace_ablation):
            assert not torch.is_grad_enabled()
            calls.append(('encode', inputs, workspace_ablation))
            return {'state': workspace_ablation}
        def decoder(self, state, *, context):
            calls.append(('decode', state, context))
            return torch.tensor([[1], [2], [3]])
    model = Model()
    for mode in ('bypass', None):
        assert mod.proposal_batch(model, 'prompt', workspace_ablation=mode) == ['repeat', 'repeat', '']
    assert calls[0] == ('encode', ['prompt'], 'bypass')
    assert calls[2] == ('encode', ['prompt'], None)
    assert calls[1][2] == calls[3][2] == {'max_new_tokens': 17, 'do_sample': False,
        'num_beams': 3, 'num_return_sequences': 3, 'return_dict_in_generate': False}


def test_case_validation_and_cli_contract(tmp_path):
    import json
    mod = runner()
    path = tmp_path / 'known.jsonl'
    path.write_text(json.dumps(case()) + '\n')
    assert mod.load_cases(path) == [case()]
    path.write_text((json.dumps(case()) + '\n') * 2)
    with pytest.raises(ValueError):
        mod.load_cases(path)
    parser = mod.parser()
    args = parser.parse_args(['--foundation', '/snapshot', '--revision', 'abc', '--cases', str(path), '--output', '/new'])
    assert args.component == 'language'
    with pytest.raises(SystemExit):
        parser.parse_args(['--foundation', '/snapshot', '--model', '/model', '--cases', str(path), '--output', '/new'])


def test_tiny_native_model_runs_all_three_modes_without_gpu(monkeypatch):
    import runpy
    from tensorcode.tools.chatbot import Chatbot
    config = runpy.run_path(str(Path(__file__).parents[1] / 'models/test_chatbot_model.py'))['tiny_config']()
    model = Chatbot(config).to(dtype=torch.bfloat16).eval().requires_grad_(False)
    # This unit test uses a tiny CPU model; only the production runner owns CUDA.
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    records = list(runner().probe_case(model, case()))
    assert [row['mode'] for row in records] == ['direct_answer_bypass',
        'declarative_proposals_bypass', 'declarative_proposals_active']
    assert all('error' not in row for row in records)
    assert [len(row['candidates']) for row in records] == [1, 3, 3]
    assert records[1]['prompt'] == records[2]['prompt']
    assert all(row['input_truncated'] for row in records)
    assert all('SECRET' not in row['prompt'] for row in records)
    assert all(row['elapsed_seconds'] >= 0 for row in records)


def test_evidence_qa_prompt_changes_only_proposal_instruction():
    mod = runner()
    direct, old = mod.prompts(case())
    new_direct, new = mod.prompts(case(), proposal_style='evidence_qa')
    assert new_direct == direct
    assert new != old
    assert 'one complete sentence' in new
    assert new.split('\n', 1)[1] == old.split('\n', 1)[1]
    assert 'SECRET' not in new
    with pytest.raises(ValueError):
        mod.prompts(case(), proposal_style='unknown')
