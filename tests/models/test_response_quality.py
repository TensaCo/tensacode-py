import copy

import pytest
import torch

from tensorcode._internal.response_quality import ResponseQualityAssessor, AXES


def tiny_config(model_type='bert', max_tokens=128):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing
    tokenizer = Tokenizer(WordLevel({'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3,
                                     'question': 4, 'evidence': 5, 'candidate': 6, 'yes': 7}, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.post_processor = TemplateProcessing(single='[CLS] $A [SEP]', special_tokens=[('[CLS]', 2), ('[SEP]', 3)])
    return {'foundation_config': {'model_type': model_type, 'vocab_size': 8, 'hidden_size': 8,
             'embedding_size': 8, 'num_hidden_layers': 1, 'num_attention_heads': 2,
             'intermediate_size': 16, 'max_position_embeddings': 128,
             'hidden_dropout_prob': 0., 'attention_probs_dropout_prob': 0.},
            'tokenizer_json': tokenizer.to_str(), 'tokenizer_special_tokens': {'pad_token': '[PAD]',
             'unk_token': '[UNK]', 'cls_token': '[CLS]', 'sep_token': '[SEP]'}, 'max_tokens': max_tokens}


INPUT = {'question': 'question', 'evidence': [{'source_id': 's1', 'text': 'evidence yes'}], 'candidate': 'yes'}
TARGET = {'support': True, 'completeness': False, 'constraints': None}


@pytest.mark.parametrize('model_type', ['bert', 'electra'])
def test_masked_axes_native_gradients(model_type):
    model = ResponseQualityAssessor(tiny_config(model_type))
    logits = model(INPUT)
    assert logits.shape == (3,)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(logits[:2], torch.tensor([1., 0.]))
    loss = model.loss(INPUT, TARGET)
    assert torch.allclose(loss, expected)
    loss.backward()
    assert model.head.weight.grad[:2].abs().sum() > 0
    assert model.head.weight.grad[2].abs().sum() == 0
    assert next(model.encoder.parameters()).grad.abs().sum() > 0


def test_strict_input_and_target_boundary():
    model = ResponseQualityAssessor(tiny_config())
    for extra in ('targets', 'rationale', 'gold_answer'):
        with pytest.raises(ValueError):
            model(dict(INPUT, **{extra: 'leak'}))
    for bad in ({}, {'support': 1, 'completeness': False, 'constraints': None}, dict.fromkeys(AXES)):
        with pytest.raises(ValueError):
            model.loss(INPUT, bad)
    duplicate = copy.deepcopy(INPUT)
    duplicate['evidence'] *= 2
    with pytest.raises(ValueError):
        model(duplicate)
    with pytest.raises(ValueError):
        ResponseQualityAssessor(dict(tiny_config(), model='obsolete'))


def test_receipt_truncation_and_acceptance():
    model = ResponseQualityAssessor(tiny_config(max_tokens=16))
    inputs = dict(INPUT, candidate='yes ' * 100)
    receipt = model.receipt(inputs)
    assert receipt['input_truncated'] is True
    assert receipt['input_token_count'] > 16
    assert receipt['source_ids'] == ['s1']
    assert not model.accepts(receipt, threshold=0.)
    short = ResponseQualityAssessor(tiny_config()).receipt(INPUT)
    assert set(short['scores']) == set(AXES)
    assert not short['input_truncated']
    assert ResponseQualityAssessor.accepts(short, threshold=0.)
    for bad in ({'scores': dict.fromkeys(AXES, 1.)}, dict(short, scores=dict.fromkeys(AXES, float('nan')))):
        with pytest.raises(ValueError):
            model.accepts(bad)


def test_calibration_invalidates_and_artifact_round_trip(tmp_path):
    model = ResponseQualityAssessor(tiny_config()).eval()
    model.fit_calibration(torch.tensor([[1., 2., 3.], [-1., -2., -3.]]),
                          [dict.fromkeys(AXES, True), dict.fromkeys(AXES, False)])
    before = model.receipt(INPUT)
    assert all(before['calibrated'].values())
    model.save_pretrained(tmp_path / 'model')
    loaded = ResponseQualityAssessor.from_pretrained(tmp_path / 'model')
    assert loaded.receipt(INPUT) == before
    assert torch.equal(model(INPUT), loaded(INPUT))
    with torch.no_grad():
        loaded.head.weight.add_(.01)
    assert not any(loaded.receipt(INPUT)['calibrated'].values())
    model.loss(INPUT, TARGET)
    assert not any(model.receipt(INPUT)['calibrated'].values())


def test_tool_trainer_checkpoint_exact_next_step(tmp_path):
    from tensorcode import training
    config = tiny_config()
    first = training.ToolTrainer(ResponseQualityAssessor(config))
    session = first.capture(INPUT, TARGET, source='test:authored')
    first.step(session)
    session.save(tmp_path / 'experience.json', operations=first.operations)
    first.save_checkpoint(tmp_path / 'resume', progress={'epoch': 1})
    expected_loss = first.step(session)
    expected = [value.detach().clone() for value in first.parameters]
    second = training.ToolTrainer(ResponseQualityAssessor(config))
    restored = training.load(tmp_path / 'experience.json', operations=second.operations)
    assert second.load_checkpoint(tmp_path / 'resume') == {'epoch': 1}
    assert second.step(restored) == expected_loss
    assert all(torch.equal(left, right) for left, right in zip(expected, second.parameters))


def test_batch_loss_is_mean_of_masked_examples_and_truncation_rejected():
    model = ResponseQualityAssessor(tiny_config())
    second = {'support': False, 'completeness': None, 'constraints': None}
    expected = (model.loss(INPUT, TARGET) + model.loss(INPUT, second)) / 2
    assert torch.allclose(model.loss([INPUT, INPUT], [TARGET, second]), expected)
    with pytest.raises(ValueError):
        model.loss([INPUT, INPUT], [TARGET])
    with pytest.raises(ValueError):
        model.loss([INPUT, INPUT], [TARGET, dict.fromkeys(AXES)])
    metadata = model.input_metadata(INPUT)
    assert metadata['source_ids'] == ['s1']
    assert metadata['input_truncated'] is False
    assert metadata['input_token_count'] == model.receipt(INPUT)['input_token_count']
    with pytest.raises(ValueError, match='truncat'):
        model.loss(dict(INPUT, candidate='yes ' * 200), TARGET)


@pytest.mark.parametrize('model_type', ['bert', 'electra'])
def test_foundation_import_owns_native_weights_and_tokenizer(tmp_path, model_type):
    from transformers import AutoModelForSequenceClassification
    from tensorcode._internal.vec.text import _native_config
    configured = ResponseQualityAssessor(tiny_config(model_type))
    native = AutoModelForSequenceClassification.from_config(_native_config(configured.config['foundation_config']))
    native.save_pretrained(tmp_path / 'foundation', safe_serialization=True)
    configured.tokenizer.save_pretrained(tmp_path / 'foundation')
    loaded = ResponseQualityAssessor.from_foundation(tmp_path / 'foundation', max_tokens=128, local_files_only=True)
    original = native.base_model.state_dict()
    assert all(torch.equal(value, original[key]) for key, value in loaded.encoder.state_dict().items())
    assert loaded.receipt(INPUT)['model']['response_quality_heads_pretrained'] is False
    loaded.save_pretrained(tmp_path / 'owned')
    restored = ResponseQualityAssessor.from_pretrained(tmp_path / 'owned')
    assert restored.receipt(INPUT) == loaded.receipt(INPUT)


def test_acceptance_requires_consistent_coverage():
    receipt = ResponseQualityAssessor(tiny_config()).receipt(INPUT)
    for malformed in (dict(receipt, input_token_count=1000), dict(receipt, max_tokens=0),
                      {key: value for key, value in receipt.items() if key != 'input_token_count'}):
        with pytest.raises(ValueError):
            ResponseQualityAssessor.accepts(malformed, threshold=0.)


def test_tokenizer_contract_and_nonfinite_receipts():
    for options in ({'json': 'override'}, {'padding_side': 'left'}, {'truncation_side': 'left'}):
        with pytest.raises(ValueError):
            ResponseQualityAssessor(dict(tiny_config(), tokenizer_options=options))
    model = ResponseQualityAssessor(tiny_config())
    model.calibrations['support'].temperature.fill_(float('nan'))
    with pytest.raises(ValueError, match='finite'):
        model.receipt(INPUT)
