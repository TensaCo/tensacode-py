"""Readout must attend to real inputs, survive artifacts, and ignore padding."""
import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import (BertConfig, BertModel, AlbertConfig, AlbertModel, RobertaConfig, RobertaModel, T5Config, T5ForConditionalGeneration,
                          PreTrainedTokenizerFast, ViTConfig, ViTModel, ViTImageProcessor)
from tensorcode.ops.vec.encode import TextEncoder, ImageEncoder
from tensorcode.ops.vec.latent import Latent, Space


@pytest.fixture(params=['bert', 't5', 'albert', 'roberta'])
def text_encoder(request, tmp_path):
    backend = Tokenizer(models.WordLevel({'[PAD]':0, '[UNK]':1, 'hello':2, 'world':3}, unk_token='[UNK]'))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='[PAD]', unk_token='[UNK]')
    if request.param == 'bert':
        native = BertModel(BertConfig(vocab_size=4, hidden_size=8, intermediate_size=16,
            num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=8,
            hidden_dropout_prob=0., attention_probs_dropout_prob=0.))
    elif request.param == 'albert':
        native = AlbertModel(AlbertConfig(vocab_size=4, hidden_size=8, embedding_size=4,
            intermediate_size=16, num_hidden_layers=1, num_hidden_groups=1,
            num_attention_heads=2, max_position_embeddings=8,
            hidden_dropout_prob=0., attention_probs_dropout_prob=0.))
    elif request.param == 'roberta':
        native = RobertaModel(RobertaConfig(vocab_size=4, hidden_size=8, intermediate_size=16,
            num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=8,
            pad_token_id=0, hidden_dropout_prob=0., attention_probs_dropout_prob=0.))
    else:
        native = T5ForConditionalGeneration(T5Config(vocab_size=4, d_model=8, d_ff=16,
            num_layers=1, num_decoder_layers=1, num_heads=2, d_kv=4, dropout_rate=0.,
            pad_token_id=0, eos_token_id=1, decoder_start_token_id=0))
    native.save_pretrained(tmp_path/'native')
    tokenizer.save_pretrained(tmp_path/'native')
    encoder = TextEncoder.from_foundation(tmp_path/'native', readout='output_encoding',
        context_space=Space('context', native.get_input_embeddings().weight.shape[-1], organization='sequence'))
    assert not any(p.is_meta for p in encoder.parameters())
    for name, weight in native.state_dict().items():
        torch.testing.assert_close(encoder.model.state_dict()[name], weight, rtol=0, atol=0)
    return encoder


def test_text_native_readout_padding_context_gradients_and_artifact(text_encoder, tmp_path):
    encoder = text_encoder
    native = encoder.model.get_encoder() if encoder.model.config.is_encoder_decoder else encoder.model
    prefix = torch.randn(2, 2, native.get_input_embeddings().weight.shape[-1], requires_grad=True)
    context = Latent(prefix, encoder.context_space, mask=torch.tensor([[True, False], [True, True]]), sources=('evidence',))
    result = encoder(['hello', 'hello world'], context={'latents':[context]})
    for row, ids in enumerate(([2], [2, 3])):
        valid = prefix[row:row+1, :row+1]
        embedded = native.get_input_embeddings()(torch.tensor([ids]))
        manual = native(inputs_embeds=torch.cat([valid, embedded, encoder.output_encoding.expand(1, 1, -1)], 1)).last_hidden_state[:, -1]
        torch.testing.assert_close(result.tensor[row:row+1], manual)
    assert result.sources == ('evidence',)
    assert result.mask.tolist() == [True, True]
    torch.testing.assert_close(encoder(['hello', 'hello world']).tensor[0], encoder('hello').tensor[0])
    encoder.tokenizer.padding_side = 'left'
    torch.testing.assert_close(encoder(['hello', 'hello world']).tensor[0], encoder('hello').tensor[0])
    assert not torch.allclose(result.tensor, encoder(['hello', 'hello world']).tensor)
    result.tensor[:, 0].sum().backward()
    assert encoder.output_encoding.grad.abs().sum() > 0
    assert native.get_input_embeddings().weight.grad.abs().sum() > 0
    assert prefix.grad[0, 0].abs().sum() > 0
    assert prefix.grad[0, 1].abs().sum() == 0
    encoder.save_pretrained(tmp_path/'owned')
    loaded = TextEncoder.from_pretrained(tmp_path/'owned')
    torch.testing.assert_close(loaded(['hello', 'hello world'], context={'latents':[context]}).tensor, result.tensor, rtol=0, atol=0)
    assert result.metadata['readout_initialization'] == 'untrained'


def test_text_position_limit_and_output_space(text_encoder):
    encoder = text_encoder
    bad = encoder.configuration()
    bad['output_space']['organization'] = 'sequence'
    with pytest.raises(ValueError, match='space'):
        TextEncoder(bad)
    if encoder.model.config.model_type != 't5':
        maximum_words = 6 if encoder.model.config.model_type == 'roberta' else 7
        encoder(' '.join(['hello'] * maximum_words))
        with pytest.raises(ValueError, match='position'):
            encoder(' '.join(['hello'] * (maximum_words + 1)))
        prefix = Latent(torch.randn(1, 1, encoder.context_space.dimensions), encoder.context_space)
        with pytest.raises(ValueError, match='position'):
            encoder(' '.join(['hello'] * maximum_words), context={'latents': [prefix]})


def test_vit_native_readout_context_gradients_and_artifact(tmp_path):
    config = ViTConfig(image_size=8, patch_size=4, hidden_size=8, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=16)
    native = ViTModel(config, add_pooling_layer=False).eval()
    native.save_pretrained(tmp_path/'native')
    ViTImageProcessor(size={'height':8, 'width':8}).save_pretrained(tmp_path/'native')
    encoder = ImageEncoder.from_foundation(tmp_path/'native', readout='output_encoding',
        output_space=Space('visual-readout', 8), context_space=Space('context', 8, organization='sequence'))
    for name, weight in native.state_dict().items():
        torch.testing.assert_close(encoder.model.state_dict()[name], weight, rtol=0, atol=0)
    pixels = torch.rand(2, 3, 8, 8, requires_grad=True)
    prefix = torch.randn(2, 2, 8, requires_grad=True)
    context = Latent(prefix, encoder.context_space, mask=torch.tensor([[True, False], [True, True]]))
    result = encoder(pixels, context={'latents':[context]})
    for row in range(2):
        hidden = torch.cat([prefix[row:row+1, :row+1], native.embeddings((pixels[row:row+1]-.5)/.5), encoder.output_encoding.expand(1, 1, -1)], 1)
        for layer in native.layers:
            hidden = layer(hidden, None)
        torch.testing.assert_close(result.tensor[row], native.layernorm(hidden)[:, -1][0])
    assert result.coordinates is None
    assert not torch.allclose(result.tensor, encoder(pixels).tensor)
    result.tensor[:, 0].sum().backward()
    assert pixels.grad.abs().sum() > 0
    assert encoder.output_encoding.grad.abs().sum() > 0
    assert prefix.grad[0, 0].abs().sum() > 0
    assert prefix.grad[0, 1].abs().sum() == 0
    encoder.save_pretrained(tmp_path/'owned')
    loaded = ImageEncoder.from_pretrained(tmp_path/'owned')
    torch.testing.assert_close(result.tensor, loaded(pixels, context={'latents':[context]}).tensor, rtol=0, atol=0)
    bad = encoder.configuration()
    bad['output_space']['organization'] = 'sequence'
    with pytest.raises(ValueError, match='space'):
        ImageEncoder(bad)
    assert result.metadata['readout_initialization'] == 'untrained'
