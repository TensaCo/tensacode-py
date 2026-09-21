"""Adapter tests use supplied fakes; they establish transport, not model ability."""
import io
import pytest
torch = pytest.importorskip("torch")
Image = pytest.importorskip("PIL.Image")
from tensorcode.integrations.local import LocalModel
from tensorcode.ops.text.messages import Message, ImagePart
from tensorcode.ops.text.model import ModelRequest


class Processor:
    def __init__(self, answer='A small cat.'):
        self.answer = answer
        self.messages = None
        self.images = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        return 'prompt'

    def __call__(self, *, text, images=None, return_tensors):
        self.images = images
        return {'input_ids': torch.tensor([[1, 2]])}

    def batch_decode(self, tokens, **kwargs):
        assert tokens.tolist() == [[3, 4]]
        return [self.answer]


class Model:
    device = torch.device('cpu')
    def eval(self):
        return self
    def generate(self, **kwargs):
        assert not torch.is_grad_enabled()
        return torch.tensor([[1, 2, 3, 4]])


def adapter(answer='A small cat.'):
    processor = Processor(answer)
    return LocalModel(Model(), processor, model_id='supplied-test-model'), processor


def test_local_text_and_image_preserve_order_without_prompt_echo():
    model, processor = adapter()
    raw = io.BytesIO()
    Image.new('RGB', (3, 2)).save(raw, format='PNG')
    output = model.complete(ModelRequest(messages=(Message('user', (
        ImagePart(data=raw.getvalue(), media_type='image/png', source_ref='photo-1'),
    )),)))
    assert output.text == 'A small cat.'
    assert processor.images[0].size == (3, 2)
    assert processor.messages[0]['content'][0]['type'] == 'image'
    assert output.provider_metadata['model_id'] == 'supplied-test-model'


def test_local_url_never_downloaded_implicitly():
    model, _ = adapter()
    with pytest.raises(ValueError, match='bytes'):
        model.complete(ModelRequest(messages=(Message('user', (
            ImagePart(url='https://example.invalid/photo.jpg'),
        )),)))


def test_structured_output_is_actual_json_or_an_error():
    model, _ = adapter('{"label": "cat"}')
    result = model.complete(ModelRequest(messages=(Message('user', 'Classify.'),),
        response_schema={'type': 'object'}, schema_name='choice'))
    assert result.structured == {'label': 'cat'}
    bad, _ = adapter('probably cat')
    with pytest.raises(ValueError, match='JSON'):
        bad.complete(ModelRequest(messages=(Message('user', 'Classify.'),),
            response_schema={'type': 'object'}))


def test_invalid_token_budget():
    with pytest.raises(ValueError):
        LocalModel(Model(), Processor(), model_id='test', max_new_tokens=0)


def test_token_limit_does_not_return_partial_answer_as_complete():
    model = LocalModel(Model(), Processor(), model_id='test', max_new_tokens=2)
    with pytest.raises(ValueError, match='token limit'):
        model.complete(ModelRequest(messages=(Message('user', 'Explain.'),)))
