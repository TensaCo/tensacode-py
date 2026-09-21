import importlib.util
from pathlib import Path
import sys

import pytest

from tensorcode.ops import llm


def _example():
    path = Path(__file__).parents[2] / "examples/image_inspection.py"
    spec = importlib.util.spec_from_file_location("image_inspection_example", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


inspect_image = _example().inspect_image


class InspectingModel:
    def __init__(self):
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        image = request.messages[1].content[0]
        return llm.ModelOutput(text=f"received {len(image.data)} image bytes")


def test_inspection_uses_real_bytes_mime_question_and_source_reference(tmp_path):
    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\nreal fixture bytes")
    model = InspectingModel()

    result = inspect_image(image_path, "What is visible?", model=model, detail="high")

    assert result.answer == "received 26 image bytes"
    assert result.media_type == "image/png"
    assert result.source_ref == f"file:{image_path.resolve()}"
    assert model.requests[0].messages[0] == llm.Message("user", "What is visible?")
    part = model.requests[0].messages[1].content[0]
    assert part == llm.ImagePart(
        data=image_path.read_bytes(),
        media_type="image/png",
        source_ref=result.source_ref,
        detail="high",
    )


def test_inspection_rejects_non_image_extension_before_calling_model(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("not an image")
    model = InspectingModel()

    with pytest.raises(ValueError, match="image MIME"):
        inspect_image(path, "Inspect this", model=model)

    assert model.requests == []


def test_inspection_requires_an_existing_regular_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        inspect_image(tmp_path / "missing.jpg", "Inspect this", model=InspectingModel())
    with pytest.raises(ValueError, match="regular file"):
        inspect_image(tmp_path, "Inspect this", model=InspectingModel())
