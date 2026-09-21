import importlib.util
from pathlib import Path
from types import SimpleNamespace
import json

import torch
from transformers import T5Config, T5ForConditionalGeneration, PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, pre_tokenizers


def test_collected_program_trains_readout_and_reloads_owned_operations(tmp_path):
    path=Path(__file__).parents[2]/'examples/output_encoding_learning.py'
    spec=importlib.util.spec_from_file_location('readout_learning',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    foundation=tmp_path/'foundation'
    native=T5ForConditionalGeneration(T5Config(vocab_size=6,d_model=8,d_ff=16,num_layers=1,
        num_decoder_layers=1,num_heads=2,d_kv=4,dropout_rate=0.,decoder_start_token_id=0,pad_token_id=0,eos_token_id=1))
    native.save_pretrained(foundation)
    tokenizer=Tokenizer(models.WordLevel({'<pad>':0,'</s>':1,'<unk>':2,'hello':3,'world':4,'answer':5},unk_token='<unk>'))
    tokenizer.pre_tokenizer=pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer,pad_token='<pad>',eos_token='</s>',unk_token='<unk>').save_pretrained(foundation)
    data=tmp_path/'data.jsonl';data.write_text(json.dumps({'text':'hello world','target':'answer'})+'\n')
    result=module.run(SimpleNamespace(foundation=str(foundation),revision='local-fixture',data=data,
        output=tmp_path/'run',device='cpu',steps=2,batch_size=1,lr=.01))
    assert result['readout_changed'] and result['bridge_changed']
    assert result['reloaded_loss_exact']
    assert result['examples']==1 and result['steps']==2
    assert result['supervision_sha256']
    # The captured loss must retain its encoder dependency, not freeze a latent boundary.
    stored=json.loads((tmp_path/'run/experience-00000.json').read_text())
    assert [call['operation'] for call in stored['calls']] == ['encode', 'objective']
    assert stored['calls'][1]['value']['children']['inputs'] == {
        'kind': 'output', 'call': 0, 'path': []}
    from tensorcode.ops.vec.encode import TextEncoder
    from tensorcode.ops.vec.decode import TextDecoder
    encoder = TextEncoder.from_pretrained(tmp_path/'run/encoder')
    decoder = TextDecoder.from_pretrained(tmp_path/'run/decoder')
    original = native.state_dict()
    for model in (encoder.model, decoder.model):
        assert model.state_dict().keys() == original.keys()
        assert all(torch.equal(value, original[key]) for key,value in model.state_dict().items())
