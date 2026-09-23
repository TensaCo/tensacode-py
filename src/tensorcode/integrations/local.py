"""Explicitly supplied local Transformers models, with no implicit model selection."""
from __future__ import annotations

import asyncio
import io
import json
from threading import Lock

from ..ops.text.messages import ImagePart, TextPart
from ..ops.text.model import ModelOutput


class LocalModel:
    """Adapt a supplied image/text generation model and processor.

    ``from_pretrained`` explicitly loads the caller's model ID or local directory.
    Image URLs are not fetched: supply image bytes. Structured answers are generated
    JSON, never repaired or assigned invented confidence. Operation-level validators
    still validate their schemas. Inference is serialized for shared model safety.
    """

    def __init__(self, model, processor, *, model_id: str, revision: str | None = None,
                 max_new_tokens: int = 128):
        if not isinstance(model_id, str) or not model_id:
            raise ValueError('model_id must identify the supplied model')
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 1:
            raise ValueError('max_new_tokens must be a positive integer')
        self.model = model.eval()
        self.processor = processor
        self.model_id = model_id
        self.revision = revision
        self.max_new_tokens = max_new_tokens
        self._lock = Lock()

    @classmethod
    def from_pretrained(cls, model_id: str, *, revision: str | None = None,
                        local_files_only: bool = True, device: str = 'cpu',
                        max_new_tokens: int = 128):
        """Load an explicit model; downloads require ``local_files_only=False``.

        Remote custom code is disabled. Install ``tensorcode[local]`` first.
        """
        from transformers import AutoModelForImageTextToText, AutoProcessor
        processor = AutoProcessor.from_pretrained(
            model_id, revision=revision, local_files_only=local_files_only,
            trust_remote_code=False)
        model = AutoModelForImageTextToText.from_pretrained(
            model_id, revision=revision, local_files_only=local_files_only,
            trust_remote_code=False).to(device)
        return cls(model, processor, model_id=model_id, revision=revision,
                   max_new_tokens=max_new_tokens)

    def configuration(self):
        """JSON description of this adapter; never includes credentials."""
        return {'model_id': self.model_id, 'revision': self.revision,
                'max_new_tokens': self.max_new_tokens}

    def complete(self, request):
        """Send one request and return a ``ModelOutput``."""
        import torch
        from PIL import Image

        messages, images = [], []
        instructions = request.instructions or ''
        if request.response_schema is not None:
            instructions += '\nReturn only a JSON object, without Markdown fences or explanation, matching this schema:\n' + json.dumps(
                dict(request.response_schema), sort_keys=True)
        if instructions:
            messages.append({'role': 'system', 'content': [{'type': 'text', 'text': instructions}]})
        for message in request.messages:
            parts = (TextPart(message.content),) if isinstance(message.content, str) else message.content
            content = []
            for part in parts:
                if isinstance(part, TextPart):
                    content.append({'type': 'text', 'text': part.text})
                elif isinstance(part, ImagePart):
                    if part.data is None:
                        raise ValueError('LocalModel requires image bytes; fetch URLs explicitly')
                    with Image.open(io.BytesIO(part.data)) as image:
                        images.append(image.convert('RGB'))
                    content.append({'type': 'image'})
                else:
                    raise TypeError('Unsupported message part')
            messages.append({'role': message.role, 'content': content})
        with self._lock, torch.inference_mode():
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            kwargs = {'text': prompt, 'return_tensors': 'pt'}
            if images:
                kwargs['images'] = images
            inputs = self.processor(**kwargs)
            inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
            output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
            generated = output[:, inputs['input_ids'].shape[1]:]
            eos = getattr(getattr(self.model, 'generation_config', None), 'eos_token_id', None)
            stop_tokens = (eos,) if isinstance(eos, int) else tuple(eos or ())
            finished = generated.shape[1] > 0 and int(generated[0, -1]) in stop_tokens
            if generated.shape[1] >= self.max_new_tokens and not finished:
                raise ValueError('Local model reached the token limit before finishing its answer')
            answer = self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
        metadata = {'model_id': self.model_id, 'revision': self.revision,
                    'backend': 'transformers', 'source': 'supplied_pretrained_model',
                    'generated_tokens': generated.shape[1], 'finish_reason': 'stop'}
        if request.response_schema is not None:
            try:
                structured = json.loads(answer)
            except json.JSONDecodeError as error:
                raise ValueError('Local model did not return valid JSON') from error
            if not isinstance(structured, dict):
                raise ValueError('Local model JSON must be an object')
            return ModelOutput(text=answer, structured=structured, provider_metadata=metadata)
        return ModelOutput(text=answer, provider_metadata=metadata)

    async def acomplete(self, request):
        """Asynchronous ``complete`` (runs the blocking call in a thread)."""
        return await asyncio.to_thread(self.complete, request)

    def complete_batch(self, requests):
        """Explicit sequential fallback; no claim of native batched generation."""
        return tuple(self.complete(request) for request in requests)
