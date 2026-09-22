import importlib
import json

import pytest
import torch
from transformers import ViTConfig, ViTImageProcessor, ViTModel
from tensorcode.ops.vec.latent import Latent, Space


def config(output='sequence'):
    return dict(model=json.loads(ViTConfig(image_size=8, patch_size=4, hidden_size=8,
        num_hidden_layers=1, num_attention_heads=2, intermediate_size=16).to_json_string()),
        processor=json.loads(ViTImageProcessor(size={'height': 8, 'width': 8}).to_json_string()),
        readout=output, output_space=Space('vision', 8, organization=output if output=='sequence' else 'feature').configuration(),
        context_space=Space('vision-context', 8, organization='sequence').configuration())


def encoder(output='sequence'):
    from tensorcode.ops.vec.encode import ImageEncoder
    return ImageEncoder(config(output))


def test_native_transformer_outputs_and_gradients():
    model = encoder().eval()
    pixels = torch.rand(2, 3, 8, 8, requires_grad=True)
    result = model(pixels)
    native = model.model(pixel_values=(pixels - .5)/.5).last_hidden_state[:, 1:]
    torch.testing.assert_close(result.tensor, native)
    assert result.mask.shape == (2, 4)
    assert result.coordinates[0].tolist() == [[2., 2.], [2., 6.], [6., 2.], [6., 6.]]
    result.tensor[..., 0].sum().backward()
    assert pixels.grad.abs().sum() > 0
    assert model.model.embeddings.patch_embeddings.projection.weight.grad.abs().sum() > 0


def test_context_attention_mask_provenance_and_full_reload(tmp_path):
    model = encoder('pooled').eval()
    pixels = torch.rand(1,3,8,8)
    tokens = torch.randn(1,2,8, requires_grad=True)
    context = Latent(tokens, Space('vision-context',8,organization='sequence'),
                     mask=torch.tensor([[True,False]]), sources=('context-source',))
    torch.testing.assert_close(model(pixels).tensor, model.model((pixels-.5)/.5).last_hidden_state[:,0])
    result = model(pixels, context={'latents':[context]})
    changed = model(pixels, context={'latents':[Latent(tokens + torch.tensor([[[0.]*8,[100.]*8]]), context.space, mask=context.mask)]})
    torch.testing.assert_close(result.tensor, changed.tensor)
    assert not torch.allclose(result.tensor,model(pixels).tensor)
    assert 'context-source' in result.sources
    result.tensor[...,0].sum().backward()
    assert tokens.grad[0,0].abs().sum()>0
    assert tokens.grad[0,1].abs().sum()==0
    model.save_pretrained(tmp_path/'vision')
    loaded = type(model).from_pretrained(tmp_path/'vision')
    torch.testing.assert_close(result.tensor,loaded(pixels,context={'latents':[context]}).tensor)
    assert not loaded.training
    loaded.train()
    assert loaded.model.training


def test_input_contract_and_local_foundation(tmp_path):
    model = encoder()
    with pytest.raises(ValueError): model(torch.full((3,8,8),2.))
    with pytest.raises(ValueError): model(torch.rand(3,9,8))
    with pytest.raises(ValueError): model(torch.rand(3,8,8),context={'surprise':1})
    native = ViTModel(ViTConfig(**config()['model']), add_pooling_layer=False).eval()
    native.save_pretrained(tmp_path/'foundation')
    ViTImageProcessor(**config()['processor']).save_pretrained(tmp_path/'foundation')
    loaded = type(model).from_foundation(tmp_path/'foundation', output_space=config()['output_space'])
    pixels = torch.rand(1,3,8,8)
    torch.testing.assert_close(loaded(pixels).tensor,native((pixels-.5)/.5).last_hidden_state[:,1:])
    assert loaded.configuration()['foundation']['repo'] == str(tmp_path/'foundation')


def test_processor_assets_and_trace_replay(tmp_path):
    from PIL import Image
    from tensorcode import trace
    model = encoder().eval()
    pixels = model.preprocess(Image.new('RGB',(16,16),color=(255,0,0)))
    assert pixels['pixel_values'].shape == (1,3,8,8)
    with trace() as session:
        result = model(pixels)
    torch.testing.assert_close(session.replay(session.ref(result)).tensor,result.tensor)
    model.save_pretrained(tmp_path/'owned')
    restored = type(model).from_pretrained(tmp_path/'owned')
    torch.testing.assert_close(restored(pixels).tensor,result.tensor)
    from tensorcode import training
    session.save(tmp_path/'trace.json', operations=model.operation_bindings())
    replayed = training.load_experience(tmp_path/'trace.json', operations=restored.operation_bindings())
    torch.testing.assert_close(replayed.replay(replayed.calls[0].output).tensor,result.tensor)
    asset=tmp_path/'owned'/'vision_processor.json'
    asset.write_text('{}')
    with pytest.raises(ValueError,match='processor'):
        type(model).from_pretrained(tmp_path/'owned')


def test_rejects_non_vit_foundation_before_loading_weights(tmp_path):
    from transformers import BertConfig
    BertConfig().save_pretrained(tmp_path)
    with pytest.raises(ValueError,match='ViT'):
        type(encoder()).from_foundation(tmp_path,output_space=config()['output_space'])


def test_rejects_incomplete_foundation_weights(tmp_path):
    from safetensors.torch import save_file
    ViTConfig(**config()['model']).save_pretrained(tmp_path)
    ViTImageProcessor(**config()['processor']).save_pretrained(tmp_path)
    save_file({'unrelated':torch.ones(1)},tmp_path/'model.safetensors')
    with pytest.raises(ValueError,match='missing'):
        type(encoder()).from_foundation(tmp_path,output_space=config()['output_space'])


def test_processed_source_provenance_and_unknown_input_keys():
    model=encoder().eval()
    output=model({'pixel_values':torch.zeros(1,3,8,8),'sources':['photo:1']})
    assert output.sources == ('photo:1',)
    with pytest.raises(ValueError,match='pixel_values'):
        model({'pixel_values':torch.zeros(1,3,8,8),'ignored':True})


def test_live_processor_configuration_survives_artifact_reload(tmp_path):
    model = encoder().eval()
    original = model.configuration()
    model.processor.image_mean = [0., .1, .2]
    model.processor.image_std = [.8, .9, 1.]
    model.processor.resample = 0
    image = torch.rand(1, 3, 8, 8)
    expected = model(image)
    assert model.configuration() != original
    model.save_pretrained(tmp_path/'updated')
    restored = type(model).from_pretrained(tmp_path/'updated')
    torch.testing.assert_close(restored(image).tensor, expected.tensor)
    from PIL import Image
    raw = Image.new('RGB', (17, 13), color=(17, 83, 199))
    torch.testing.assert_close(restored.preprocess(raw)['pixel_values'],
                               model.preprocess(raw)['pixel_values'])


def test_encoder_uses_output_space_and_readout_contract():
    from tensorcode.ops.vec.encode import ImageEncoder
    settings=config()
    model=ImageEncoder(settings)
    assert model.output_space==Space(**settings['output_space'])
    assert model.readout=='sequence'


def test_context_prefix_order_native_positions_and_validation():
    model=encoder().eval()
    pixels=torch.rand(1,3,8,8)
    space=model.context_space
    first=Latent(torch.randn(1,1,8),space,sources=('first',))
    second=Latent(torch.randn(1,2,8),space,mask=torch.tensor([[True,False]]),sources=('second',))
    observed=[]
    handle=model.model.layers[0].register_forward_pre_hook(lambda module,args: observed.append(args[0].detach().clone()))
    result=model(pixels,context={'latents':[first,second]})
    handle.remove()
    expected=torch.cat([first.tensor,second.tensor.masked_fill(~second.mask[...,None],0),model.model.embeddings((pixels-.5)/.5)],1)
    torch.testing.assert_close(observed[0],expected)
    assert result.tensor.shape==(1,4,8)
    assert result.sources==('first','second')
    with pytest.raises(ValueError,match='batch'):
        model(pixels,context={'latents':[Latent(torch.randn(2,1,8),space)]})
    with pytest.raises(ValueError,match='space'):
        model(pixels,context={'latents':[Latent(first.tensor,Space('wrong',8,organization='sequence'))]})
    without_context=type(model)({**config(),'context_space':None})
    with pytest.raises(ValueError,match='context_space'):
        without_context(pixels,context={'latents':[first]})
    with pytest.raises(ValueError,match='context_space'):
        type(model)({**config(),'context_space':Space('bad-width',3).configuration()})


@pytest.mark.parametrize('key,value',[('space',{}),('output','sequence')])
def test_vision_rejects_legacy_constructor_keys(key,value):
    from tensorcode.ops.vec.encode import ImageEncoder
    with pytest.raises(ValueError,match='output_space|readout'):
        ImageEncoder({**config(),key:value})
