import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, T5Config, T5ForConditionalGeneration, BertConfig, BertModel


def tokenizer():
    backend = Tokenizer(models.WordLevel({'[PAD]':0,'[UNK]':1,'hello':2,'world':3,'prefix':4,'answer':5}, unk_token='[UNK]'))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='[PAD]', unk_token='[UNK]')


@pytest.fixture
def foundation(tmp_path):
    path = tmp_path / 'native'
    model = T5ForConditionalGeneration(T5Config(vocab_size=6,d_model=8,d_ff=16,num_layers=1,num_decoder_layers=1,num_heads=2,dropout_rate=0.,decoder_start_token_id=0,eos_token_id=5,pad_token_id=0))
    model.save_pretrained(path)
    tokenizer().save_pretrained(path)
    return path


def test_encoder_native_context_padding_and_offline(foundation,tmp_path):
    from tensorcode.ops.vec.encode import TextEncoder
    encoder = TextEncoder.from_foundation(foundation)
    value = encoder(['hello world','hello'])
    batch = encoder.tokenizer(['hello world','hello'],padding=True,return_tensors='pt')
    expected = encoder.model.get_encoder()(input_ids=batch['input_ids'],attention_mask=batch['attention_mask']).last_hidden_state
    torch.testing.assert_close(value.tensor,expected)
    assert value.mask.tolist() == [[True,True],[True,False]]
    encoder.save_pretrained(tmp_path/'owned')
    restored = TextEncoder.from_pretrained(tmp_path/'owned',local_files_only=True)
    torch.testing.assert_close(restored(['hello world','hello']).tensor,value.tensor)


def test_decoder_runs_native_encoder_identity_and_loss(foundation,tmp_path):
    import json
    from tensorcode._internal.training.persistence import bindings
    from tensorcode.ops.vec.decode import TextDecoder
    from tensorcode.ops.vec.latent import Latent, Space
    decoder = TextDecoder.from_foundation(foundation,input_space=Space('arbitrary',3,organization='sequence'),generation={'max_new_tokens':3})
    saved_bindings = bindings(decoder.operation_bindings())
    assert saved_bindings['objective']['configuration']['type'] == 'tensorcode.ops.vec.decode.TextDecoder.objective'
    assert 'tensorcode._internal.vec' not in json.dumps(saved_bindings)
    x = torch.randn(2,2,3,requires_grad=True)
    value = Latent(x,decoder.input_space,mask=torch.tensor([[True,True],[True,False]]))
    calls=[]
    handle=decoder.model.get_encoder().register_forward_hook(lambda *args: calls.append(1))
    output=decoder(value)
    assert len(output)==2 and calls
    handle.remove()
    decoder.loss(value,['answer','hello']).backward()
    assert x.grad is not None and x.grad.abs().sum()>0
    assert decoder.projection.weight.grad.abs().sum()>0
    assert decoder.model.encoder.block[0].layer[0].SelfAttention.q.weight.grad.abs().sum()>0
    decoder.save_pretrained(tmp_path/'owned')
    restored=TextDecoder.from_pretrained(tmp_path/'owned')
    assert restored(value)==output
    with pytest.raises(ValueError,match='space'):
        restored(Latent(x,Space('wrong',3,organization='sequence')))
    native=TextDecoder.from_foundation(foundation,input_space=decoder.native_input_space,bridge='identity',generation={'max_new_tokens':3})
    tokens=native.tokenizer(['hello world','hello'],padding=True,return_tensors='pt')
    embeds=native.model.get_input_embeddings()(tokens['input_ids'])
    baseline=native.model.generate(input_ids=tokens['input_ids'],attention_mask=tokens['attention_mask'],max_new_tokens=3)
    assert native(Latent(embeds,native.input_space,mask=tokens['attention_mask'].bool())) == native.tokenizer.batch_decode(baseline,skip_special_tokens=True)


def test_bert_mean_masks_padding_and_rejects_target_context(tmp_path):
    from tensorcode.ops.vec.encode import TextEncoder
    path=tmp_path/'bert'
    BertModel(BertConfig(vocab_size=6,hidden_size=8,intermediate_size=16,num_hidden_layers=1,num_attention_heads=2,hidden_dropout_prob=0.,attention_probs_dropout_prob=0.)).save_pretrained(path)
    tokenizer().save_pretrained(path)
    encoder=TextEncoder.from_foundation(path,readout='pooled')
    batch=encoder(['hello','hello world'])
    torch.testing.assert_close(batch.tensor[0],encoder('hello').tensor[0])
    with pytest.raises(ValueError,match='context'):
        encoder('hello',context={'targets':'answer'})


def test_decoder_context_masks_and_modes(foundation):
    from tensorcode.ops.vec.decode import TextDecoder
    from tensorcode.ops.vec.latent import Latent,Space
    decoder=TextDecoder.from_foundation(foundation,input_space=Space('features',3,organization='sequence'),generation={'max_new_tokens':2})
    prefix=Latent(torch.randn(1,2,3),decoder.input_space,mask=torch.tensor([[True,False]]))
    value=Latent(torch.randn(1,1,3),decoder.input_space)
    concat=Latent(torch.cat([prefix.tensor,value.tensor],1),decoder.input_space,mask=torch.tensor([[True,False,True]]))
    torch.testing.assert_close(decoder.loss(value,'answer',context={'latents':[prefix]}),decoder.loss(concat,'answer'))
    poisoned=Latent(prefix.tensor.clone(),prefix.space,mask=prefix.mask)
    poisoned.tensor[:,1]=1e5
    torch.testing.assert_close(decoder.loss(value,'answer',context={'latents':[prefix]}),decoder.loss(value,'answer',context={'latents':[poisoned]}))
    decoder.train()
    decoder.model.encoder.eval()
    modes=[m.training for m in decoder.model.modules()]
    decoder(value)
    assert [m.training for m in decoder.model.modules()]==modes
    with pytest.raises(ValueError,match='context'):
        decoder(value,context={'targets':'answer'})
    with pytest.raises(ValueError,match='identity'):
        TextDecoder.from_foundation(foundation,input_space=Space('fake',8,organization='sequence'),bridge='identity')


def test_text_decoder_native_embedding_and_training_replay(foundation,tmp_path):
    from tensorcode.ops.vec.decode import TextDecoder
    from tensorcode.ops.vec.latent import Space, Latent
    from tensorcode.training import Trainer
    from tensorcode.training import load_experience
    decoder=TextDecoder.from_foundation(foundation,input_space=Space('unused',8,organization='sequence'))
    native=TextDecoder.from_foundation(foundation,input_space=decoder.native_input_space,bridge='identity')
    latent=native.embed_text(['hello world','hello'])
    assert latent.mask.tolist()==[[True,True],[True,False]]
    trainer=Trainer.from_tool(native)
    session=trainer.capture(latent,['answer','hello'],source='fixture')
    session.save(tmp_path/'experience.json',operations=trainer.operations,codecs={'latent':Latent,'space':Space})
    native.save_pretrained(tmp_path/'model')
    restored=TextDecoder.from_pretrained(tmp_path/'model')
    next_trainer=Trainer.from_tool(restored)
    loaded=load_experience(tmp_path/'experience.json',operations=next_trainer.operations,codecs={'latent':Latent,'space':Space})
    before=restored.model.encoder.block[0].layer[0].SelfAttention.q.weight.detach().clone()
    next_trainer.step(loaded)
    assert not torch.equal(before,restored.model.encoder.block[0].layer[0].SelfAttention.q.weight)


def test_foundation_requires_complete_weights(foundation):
    from safetensors.torch import load_file,save_file
    from tensorcode.ops.vec.encode import TextEncoder
    file=foundation/'model.safetensors'
    tensors=load_file(file)
    del tensors['encoder.block.0.layer.0.SelfAttention.q.weight']
    save_file(tensors,file,metadata={'format':'pt'})
    with pytest.raises(ValueError,match='missing'):
        TextEncoder.from_foundation(foundation)


def test_foundation_bridge_matches_native_dtype(foundation):
    from tensorcode.ops.vec.decode import TextDecoder
    from tensorcode.ops.vec.latent import Space,Latent
    decoder=TextDecoder.from_foundation(foundation,input_space=Space('input',3,organization='sequence'),dtype=torch.float64)
    assert decoder.projection.weight.dtype==torch.float64
    assert torch.isfinite(decoder.loss(Latent(torch.randn(1,2,3),decoder.input_space),'answer'))


def test_foundation_generation_defaults_saved_but_sampling_explicit(foundation,tmp_path):
    import json
    from tensorcode.ops.vec.decode import TextDecoder
    from tensorcode.ops.vec.latent import Space
    path=foundation/'generation_config.json'
    config=json.loads(path.read_text())
    config['do_sample']=True
    config['eos_token_id']=3
    path.write_text(json.dumps(config))
    model=TextDecoder.from_foundation(foundation,input_space=Space('input',3))
    assert model.generation['do_sample'] is False
    model.save_pretrained(tmp_path/'saved')
    restored=TextDecoder.from_pretrained(tmp_path/'saved')
    assert restored.model.generation_config.eos_token_id==3


def test_live_tokenizer_and_generation_settings_survive_artifact(foundation,tmp_path):
    from tensorcode.ops.vec.encode import TextEncoder
    from tensorcode.ops.vec.decode import TextDecoder
    from tensorcode.ops.vec.latent import Space
    encoder=TextEncoder.from_foundation(foundation)
    encoder.tokenizer.padding_side='left'
    encoder.tokenizer.clean_up_tokenization_spaces=True
    before=encoder(['hello world','hello'])
    config=encoder.configuration()
    encoder('hello')
    assert encoder.configuration()==config
    encoder.save_pretrained(tmp_path/'encoder')
    restored=TextEncoder.from_pretrained(tmp_path/'encoder')
    assert restored.tokenizer.padding_side=='left'
    assert restored.tokenizer.clean_up_tokenization_spaces is True
    torch.testing.assert_close(before.tensor,restored(['hello world','hello']).tensor)
    decoder=TextDecoder.from_foundation(foundation,input_space=Space('input',3))
    decoder.generation['max_new_tokens']=7
    decoder.model.generation_config.eos_token_id=3
    decoder.tokenizer.padding_side='left'
    decoder.save_pretrained(tmp_path/'decoder')
    decoded=TextDecoder.from_pretrained(tmp_path/'decoder')
    assert decoded.generation['max_new_tokens']==7
    assert decoded.model.generation_config.eos_token_id==3
    assert decoded.tokenizer.padding_side=='left'


def test_tokenizer_cleanup_native_contract():
    from tensorcode._internal.vec.text import _tokenizer,_tokenizer_config
    backend=Tokenizer(models.WordLevel({'[UNK]':0,'hello':1,',':2},unk_token='[UNK]'))
    original=PreTrainedTokenizerFast(tokenizer_object=backend,unk_token='[UNK]',clean_up_tokenization_spaces=True)
    restored=_tokenizer(_tokenizer_config(original))
    assert original.decode([1,2])=='hello,'
    assert restored.decode([1,2])=='hello,'


def test_training_capture_context_envelope_fresh_replay(foundation,tmp_path):
    from tensorcode.ops.vec.decode import TextDecoder
    from tensorcode.ops.vec.latent import Space,Latent
    from tensorcode.training import Trainer,load_experience
    decoder=TextDecoder.from_foundation(foundation,input_space=Space('input',3,organization='sequence'))
    trainer=Trainer.from_tool(decoder)
    value=Latent(torch.randn(1,2,3),decoder.input_space)
    prefix=Latent(torch.randn(1,1,3,requires_grad=True),decoder.input_space)
    envelope={'value':value,'context':{'latents':[prefix]}}
    loss=decoder.training_operation({'inputs':envelope,'targets':'answer'})
    loss.backward()
    assert prefix.tensor.grad.abs().sum()>0
    session=trainer.capture(envelope,'answer',source='fixture')
    session.save(tmp_path/'context.json',operations=trainer.operations,codecs={'latent':Latent,'space':Space})
    decoder.save_pretrained(tmp_path/'model')
    restored=TextDecoder.from_pretrained(tmp_path/'model')
    restarted=Trainer.from_tool(restored)
    experience=load_experience(tmp_path/'context.json',operations=restarted.operations,codecs={'latent':Latent,'space':Space})
    assert torch.isfinite(torch.as_tensor(restarted.step(experience)))
    with pytest.raises(ValueError,match='envelope'):
        trainer.capture({**envelope,'targets':'leaked'},'answer',source='fixture')
    with pytest.raises(ValueError,match='context'):
        trainer.capture({'value':value,'context':{'targets':'leaked'}},'answer',source='fixture')


@pytest.mark.parametrize("scale_outputs",[False,True])
def test_untied_foundation_keeps_distinct_lm_head(foundation,tmp_path,scale_outputs):
    from tensorcode.ops.vec.decode import TextDecoder
    from tensorcode.ops.vec.latent import Space
    config=T5Config(vocab_size=6,d_model=8,d_ff=16,num_layers=1,num_decoder_layers=1,num_heads=2,dropout_rate=0.,decoder_start_token_id=0,eos_token_id=5,pad_token_id=0,tie_word_embeddings=False)
    config.tie_word_embeddings=False
    config.scale_decoder_outputs=scale_outputs
    native=T5ForConditionalGeneration(config)
    with torch.no_grad():
        native.shared.weight.fill_(0.25)
        native.lm_head.weight.fill_(0.75)
    native.eval()
    native.save_pretrained(foundation)
    decoder=TextDecoder.from_foundation(foundation,input_space=Space('input',3))
    assert decoder.model.config.tie_word_embeddings is False
    decoder.save_pretrained(tmp_path/'owned')
    restored=TextDecoder.from_pretrained(tmp_path/'owned')
    assert restored.model.config.tie_word_embeddings is False
    assert restored.model.config.scale_decoder_outputs is scale_outputs
    for key,tensor in native.state_dict().items():
        torch.testing.assert_close(restored.model.state_dict()[key],tensor)
    inputs={'input_ids':torch.tensor([[2,3]]),'decoder_input_ids':torch.tensor([[0,2]])}
    torch.testing.assert_close(restored.model(**inputs).logits,native(**inputs).logits)
    assert restored.model.shared.weight.data_ptr()!=restored.model.lm_head.weight.data_ptr()
    torch.testing.assert_close(restored.model.shared.weight,torch.full_like(restored.model.shared.weight,0.25))
    torch.testing.assert_close(restored.model.lm_head.weight,torch.full_like(restored.model.lm_head.weight,0.75))


def test_text_latent_prefix_readout_mask_and_gradient(foundation):
    from tensorcode.ops.vec.encode import TextEncoder
    from tensorcode.ops.vec.latent import Latent, Space
    space = Space('explicit-native-embeddings',8,organization='sequence')
    encoder = TextEncoder.from_foundation(foundation,readout='sequence',context_space=space)
    raw = torch.randn(2,2,8,requires_grad=True)
    prefix = Latent(raw,space,mask=torch.tensor([[True,False],[True,True]]),sources=('prefix:1',))
    texts = ['hello world','hello']
    result = encoder(texts,context={'latents':[prefix]})
    tokens = encoder.tokenizer(texts,padding=True,return_tensors='pt')
    embeds = encoder.model.get_input_embeddings()(tokens['input_ids'])
    expected = torch.zeros_like(result.tensor)
    for row in range(len(texts)):
        valid_text = tokens['attention_mask'][row].bool()
        native_inputs = torch.cat([raw[row][prefix.mask[row]], embeds[row][valid_text]])[None]
        native = encoder.model.get_encoder()(inputs_embeds=native_inputs).last_hidden_state
        expected[row, valid_text] = native[0, prefix.mask[row].sum():]
    torch.testing.assert_close(result.tensor,expected)
    assert result.mask.tolist()==[[True,True],[True,False]]
    assert result.sources==('prefix:1',)
    result.tensor[...,0].sum().backward()
    assert raw.grad[0,0].abs().sum()>0
    assert raw.grad[0,1].abs().sum()==0
    pooled_config={**encoder.configuration(),'readout':'pooled','output_space':Space('pooled',8).configuration()}
    pooled=TextEncoder(pooled_config).eval()
    pooled.model.load_state_dict(encoder.model.state_dict())
    pooled_result=pooled(texts,context={'latents':[prefix]})
    torch.testing.assert_close(pooled_result.tensor,(expected*result.mask[...,None]).sum(1)/result.mask.sum(1,keepdim=True))
    with pytest.raises(ValueError,match='batch'):
        encoder('hello',context={'latents':[prefix]})
    with pytest.raises(ValueError,match='space'):
        encoder(texts,context={'latents':[Latent(raw,Space('wrong',8,organization='sequence'))]})
    undeclared=TextEncoder.from_foundation(foundation)
    with pytest.raises(ValueError,match='context_space'):
        undeclared(texts,context={'latents':[prefix]})
    with pytest.raises(ValueError,match='context'):
        encoder('hello',context={'texts':['prefix']})
    with pytest.raises(ValueError,match='context_space'):
        TextEncoder({**encoder.configuration(),'context_space':Space('wrong-width',3).configuration()})


@pytest.mark.parametrize('key,value',[('space',{}),('output','sequence'),('pooling','mean')])
def test_text_rejects_legacy_constructor_keys(foundation,key,value):
    from tensorcode.ops.vec.encode import TextEncoder
    encoder=TextEncoder.from_foundation(foundation)
    with pytest.raises(ValueError,match='output_space|readout'):
        TextEncoder({**encoder.configuration(),key:value})


def test_albert_context_uses_input_embedding_width(tmp_path):
    from transformers import AlbertConfig, AlbertModel
    from tensorcode.ops.vec.encode import TextEncoder
    from tensorcode.ops.vec.latent import Latent, Space
    path=tmp_path/'albert'
    AlbertModel(AlbertConfig(vocab_size=6,embedding_size=4,hidden_size=8,intermediate_size=16,num_hidden_layers=1,num_hidden_groups=1,num_attention_heads=2,hidden_dropout_prob=0.,attention_probs_dropout_prob=0.)).save_pretrained(path)
    tokenizer().save_pretrained(path)
    space=Space('albert-input',4,organization='sequence')
    encoder=TextEncoder.from_foundation(path,context_space=space)
    assert encoder.output_space.dimensions==8
    raw=torch.randn(1,2,4,requires_grad=True)
    prefix=Latent(raw,space,mask=torch.tensor([[True,False]]))
    output=encoder('hello world',context={'latents':[prefix]})
    tokens=encoder.tokenizer(['hello world'],return_tensors='pt')
    inputs=encoder.model.get_input_embeddings()(tokens['input_ids'])
    expected=encoder.model(inputs_embeds=torch.cat([raw[:, :1],inputs],1)).last_hidden_state[:,1:]
    torch.testing.assert_close(output.tensor,expected)
    output.tensor[...,0].sum().backward()
    assert raw.grad[0,0].abs().sum()>0
    assert raw.grad[0,1].abs().sum()==0
    with pytest.raises(ValueError,match='context_space'):
        TextEncoder.from_foundation(path,context_space=Space('wrong-hidden-width',8,organization='sequence'))
    encoder.save_pretrained(tmp_path/'owned-albert')
    restored=TextEncoder.from_pretrained(tmp_path/'owned-albert')
    torch.testing.assert_close(output.tensor,restored('hello world',context={'latents':[prefix]}).tensor)


def test_encoder_uses_actual_untied_encoder_embeddings(foundation):
    from torch import nn
    from tensorcode.ops.vec.encode import TextEncoder
    from tensorcode.ops.vec.latent import Latent, Space
    space=Space('encoder-input',8,organization='sequence')
    encoder=TextEncoder.from_foundation(foundation,context_space=space)
    encoder.model.encoder.embed_tokens=nn.Embedding(6,8)
    prefix=Latent(torch.randn(1,1,8),space)
    tokens=encoder.tokenizer(['hello world'],return_tensors='pt')
    embeds=encoder.model.encoder.get_input_embeddings()(tokens['input_ids'])
    expected=encoder.model.encoder(inputs_embeds=torch.cat([prefix.tensor,embeds],1),attention_mask=torch.ones(1,3)).last_hidden_state[:,1:]
    torch.testing.assert_close(encoder('hello world',context={'latents':[prefix]}).tensor,expected)


def test_decoder_embed_text_uses_actual_encoder_embeddings(foundation):
    from torch import nn
    from tensorcode.ops.vec.decode import TextDecoder
    from tensorcode.ops.vec.latent import Space
    decoder=TextDecoder.from_foundation(foundation,input_space=Space('input',3))
    decoder.model.encoder.embed_tokens=nn.Embedding(6,8)
    tokens=decoder.tokenizer(['hello world'],return_tensors='pt')
    expected=decoder.model.encoder.get_input_embeddings()(tokens['input_ids'])
    torch.testing.assert_close(decoder.embed_text('hello world').tensor,expected)


def test_decoder_padding_holes_preserve_loss_and_generation_scores(foundation):
    from tensorcode.ops.vec.decode import TextDecoder
    from tensorcode.ops.vec.latent import Latent, Space
    torch.manual_seed(4)
    decoder = TextDecoder.from_foundation(
        foundation, input_space=Space('padding', 3, organization='sequence'),
        generation={'max_new_tokens': 3})
    value = Latent(torch.randn(1, 2, 3), decoder.input_space)
    prefix = Latent(torch.randn(1, 1, 3), decoder.input_space)
    padded = Latent(torch.cat([prefix.tensor, torch.randn(1, 20, 3)], 1),
                    decoder.input_space, mask=torch.tensor([[True] + [False] * 20]))
    torch.testing.assert_close(
        decoder.loss(value, 'answer', context={'latents': [prefix]}),
        decoder.loss(value, 'answer', context={'latents': [padded]}))
    # Tiny random models can decode identical empty strings despite changed
    # probabilities, so compare actual autoregressive scores as well.
    outputs = []
    for item in (prefix, padded):
        embeddings, mask = decoder._inputs(value, {'latents': [item]})
        outputs.append(decoder.model.generate(
            inputs_embeds=embeddings, attention_mask=mask, max_new_tokens=3,
            return_dict_in_generate=True, output_scores=True))
    torch.testing.assert_close(outputs[0].sequences, outputs[1].sequences)
    for first, second in zip(outputs[0].scores, outputs[1].scores):
        torch.testing.assert_close(first, second)
    assert decoder(value, context={'latents': [prefix]}) == decoder(value, context={'latents': [padded]})


def test_decoder_packs_varied_rows_and_preserves_only_valid_gradients(foundation):
    from tensorcode.ops.vec.decode import TextDecoder
    from tensorcode.ops.vec.latent import Latent, Space
    decoder = TextDecoder.from_foundation(
        foundation, input_space=Space('padding', 3, organization='sequence'))
    prefix_tensor = torch.randn(2, 4, 3, requires_grad=True)
    main_tensor = torch.randn(2, 4, 3, requires_grad=True)
    prefix_mask = torch.tensor([[False, True, False, True], [True, False, False, False]])
    main_mask = torch.tensor([[True, False, True, False], [False, False, True, False]])
    prefix = Latent(prefix_tensor, decoder.input_space, mask=prefix_mask)
    value = Latent(main_tensor, decoder.input_space, mask=main_mask)
    embeddings, mask = decoder._inputs(value, {'latents': [prefix]})
    assert mask.tolist() == [[True, True, True, True], [True, True, False, False]]
    labels = torch.tensor([[5], [5]])
    batch = decoder.model(inputs_embeds=embeddings, attention_mask=mask, labels=labels)
    for row in range(2):
        compact = Latent(torch.cat([prefix_tensor[row][prefix_mask[row]],
                                   main_tensor[row][main_mask[row]]])[None], decoder.input_space)
        native_embeddings, native_mask = decoder._inputs(compact, None)
        single = decoder.model(inputs_embeds=native_embeddings, attention_mask=native_mask, labels=labels[:1])
        torch.testing.assert_close(batch.logits[row], single.logits[0], rtol=1e-5, atol=1e-6)
    batch.loss.backward()
    for tensor, valid in ((prefix_tensor, prefix_mask), (main_tensor, main_mask)):
        assert torch.count_nonzero(tensor.grad[~valid]) == 0
        assert torch.isfinite(tensor.grad).all()
        assert tensor.grad[valid].abs().sum() > 0
    assert decoder.projection.weight.grad.abs().sum() > 0
    assert decoder.model.encoder.block[0].layer[0].SelfAttention.q.weight.grad.abs().sum() > 0


@pytest.mark.parametrize('readout', ['sequence', 'pooled'])
def test_encoder_prefix_padding_is_position_and_gradient_invariant(foundation, readout):
    from tensorcode.ops.vec.encode import TextEncoder
    from tensorcode.ops.vec.latent import Latent, Space
    space = Space('native-context', 8, organization='sequence')
    encoder = TextEncoder.from_foundation(foundation, readout=readout, context_space=space)
    encoder.tokenizer.padding_side = 'left'
    raw = torch.randn(2, 4, 8, requires_grad=True)
    keep = torch.tensor([[False, True, False, True], [True, False, False, False]])
    prefix = Latent(raw, space, mask=keep)
    batch = encoder(['hello world', 'hello'], context={'latents': [prefix]})
    for row, text in enumerate(['hello world', 'hello']):
        compact = Latent(raw[row][keep[row]][None], space)
        single = encoder(text, context={'latents': [compact]})
        actual = batch.tensor[row][batch.mask[row]] if readout == 'sequence' else batch.tensor[row]
        expected = single.tensor[0][single.mask[0]] if readout == 'sequence' else single.tensor[0]
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    if readout == 'sequence':
        assert batch.mask.tolist() == [[True, True], [False, True]]
        assert torch.count_nonzero(batch.tensor[~batch.mask]) == 0
    batch.tensor.square().sum().backward()
    assert torch.count_nonzero(raw.grad[~keep]) == 0
    assert torch.isfinite(raw.grad).all()
    assert raw.grad[keep].abs().sum() > 0


@pytest.mark.parametrize('kind', ['decoder', 'sequence', 'pooled'])
def test_masked_text_conditioning_survives_artifact_and_durable_replay(foundation, tmp_path, kind):
    from tensorcode import trace, training
    from tensorcode.ops.vec.decode import TextDecoder
    from tensorcode.ops.vec.encode import TextEncoder
    from tensorcode.ops.vec.latent import Latent, Space
    space = Space('conditioning', 3 if kind == 'decoder' else 8, organization='sequence')
    if kind == 'decoder':
        model = TextDecoder.from_foundation(foundation, input_space=space)
        value = Latent(torch.randn(1, 2, 3), space)
    else:
        model = TextEncoder.from_foundation(foundation, context_space=space, readout=kind)
        value = 'hello world'
    raw = torch.randn(1, 1, space.dimensions)
    prefix = Latent(torch.cat([raw, torch.full((1, 20, space.dimensions), 1e30)], 1),
                    space, mask=torch.tensor([[True] + [False] * 20]))
    compact = Latent(raw, space)
    if kind == 'decoder':
        trainer = training.Trainer.from_tool(model)
        session = trainer.capture({'value': value, 'context': {'latents': [prefix]}},
                                  'answer', source='padding regression fixture')
        expected = model.loss(value, 'answer', context={'latents': [compact]})
    else:
        with trace() as session:
            result = model(value, context={'latents': [prefix]})
        expected = model(value, context={'latents': [compact]}).tensor
        torch.testing.assert_close(result.tensor, expected)
    model.save_pretrained(tmp_path/'model')
    restored = type(model).from_pretrained(tmp_path/'model')
    session.save(tmp_path/'trace.json', operations=model.operation_bindings(),
                 codecs={'latent': Latent, 'space': Space})
    replayed = training.load_experience(tmp_path/'trace.json', operations=restored.operation_bindings(),
                             codecs={'latent': Latent, 'space': Space})
    actual = replayed.replay(replayed.calls[-1].output)
    torch.testing.assert_close(actual if kind == 'decoder' else actual.tensor, expected)
    if kind == 'decoder':
        assert torch.isfinite(torch.as_tensor(training.Trainer.from_tool(restored).step(replayed)))
