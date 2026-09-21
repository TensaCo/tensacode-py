"""Small authored fixtures test provenance and isolation, not NLI ability."""
import importlib.util
from pathlib import Path

import pytest
import torch

spec = importlib.util.spec_from_file_location('compare_cognitive_verifiers', Path(__file__).parents[2] / 'examples/compare_cognitive_verifiers.py')
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


def test_semantic_mapping_uses_native_order_and_rejects_ambiguous_labels():
    mapping = example.sibling('train_verifier').label_mapping
    assert mapping({'0': 'entailment', '1': 'neutral', '2': 'contradiction'}) == {'support': 0, 'unknown': 1, 'contradiction': 2}
    with pytest.raises(ValueError, match='label mapping'):
        mapping({'0': 'LABEL_0', '1': 'LABEL_1', '2': 'LABEL_2'})


def test_historical_recovery_keeps_sources_but_excludes_targets_from_inputs():
    evidence = [{'id': 'e1', 'source_id': 'original-document', 'text': 'actual source text', 'irrelevant': 'discard'}]
    report = {'real_data': {'records': [{'id': 'q1', 'question': 'What?', 'target': 'gold answer',
              'answer': 'old generated response', 'receipt': {'cognition': {'evidence': evidence}}}]}}
    case, = example.recover_cases(report)
    assert case['target'] == 'gold answer'
    assert example.case_inputs(case) == {'question': 'What?', 'evidence': [{'id': 'e1', 'source_id': 'original-document', 'text': 'actual source text'}]}
    assert 'gold answer' not in str(example.case_inputs(case))
    assert 'old generated response' not in str(case)
    assert 'development' in case['source_kind']


def test_unchanged_detection_catches_edits_additions_and_removals():
    before = {'foundation.weight': 'a', 'investigator.verifier.model.weight': 'old'}
    example.assert_unchanged(before, {**before, 'investigator.verifier.model.weight': 'new'})
    for after in ({**before, 'foundation.weight': 'b'}, {**before, 'memory': 'c'}, {'investigator.verifier.model.weight': 'new'}):
        with pytest.raises(ValueError, match='non-verifier tensors changed'):
            example.assert_unchanged(before, after)


def test_tensor_digest_tracks_values_shapes_and_dtypes():
    model = torch.nn.Linear(2, 2, bias=False)
    before = example.tensor_digests(model)
    assert before == example.tensor_digests(model)
    with torch.no_grad():
        model.weight[0, 0] += 1
    assert before != example.tensor_digests(model)
    changed = example.tensor_digests(model)
    assert changed != example.tensor_digests(model.double())


def test_only_verifier_configuration_is_excluded_from_comparison():
    config = {'cognition': {'policy': {'min_support': .9}, 'investigator': {'generator': {'x': 1}, 'verifier_labels': {'support': 0}}}}
    filtered = example.without_verifier(config)
    assert filtered == {'cognition': {'policy': {'min_support': .9}, 'investigator': {'generator': {'x': 1}}}}
    assert 'verifier_labels' in config['cognition']['investigator']


def test_native_integer_label_config_is_normalized_for_complete_tool(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import BertConfig, BertForSequenceClassification, PreTrainedTokenizerFast
    from tensorcode._internal.pretrained import PretrainedTool
    native = BertForSequenceClassification(BertConfig(
        vocab_size=4, hidden_size=8, intermediate_size=16, num_hidden_layers=1,
        num_attention_heads=2, num_labels=3,
        id2label={0: 'entailment', 1: 'neutral', 2: 'contradiction'})).eval()
    assert set(native.config.to_dict()['id2label']) == {0, 1, 2}
    native.save_pretrained(tmp_path)
    backend = Tokenizer(models.WordLevel({'[PAD]': 0, '[UNK]': 1, 'alpha': 2, 'beta': 3}, unk_token='[UNK]'))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='[PAD]', unk_token='[UNK]')
    tokenizer.save_pretrained(tmp_path)
    verifier = example.make_verifier(tmp_path, 'authored/tiny-bert', 'test-revision')
    configuration = verifier.configuration()
    assert PretrainedTool._validated_config(configuration) == configuration
    assert set(configuration['verifier_config']['id2label']) == {'0', '1', '2'}
    assert verifier.labels == {'support': 0, 'unknown': 1, 'contradiction': 2}
    assert example.tensor_digests(native) == example.tensor_digests(verifier.model)
    assert float(verifier.calibration.temperature) == 1.
    assert not bool(verifier.calibration.calibrated)
