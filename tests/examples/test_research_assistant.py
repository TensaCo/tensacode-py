import importlib.util
from pathlib import Path
import sys

import pytest

from tensorcode.ops import text as text_ops


def _example():
    path = Path(__file__).parents[2] / "examples/research_assistant.py"
    spec = importlib.util.spec_from_file_location("research_assistant_example", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


research = _example().research


class ScriptedResearchModel:
    def __init__(self, choices, answer="Grounded answer [doc-0001]"):
        self.choices = list(choices)
        self.answer = answer
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        if request.response_schema is not None:
            return text_ops.ModelOutput(
                structured={
                    "choice": self.choices.pop(0),
                    "distribution": None,
                    "confidence": None,
                    "abstained": False,
                }
            )
        return text_ops.ModelOutput(text=self.answer)


def test_research_loop_reads_only_selected_document_and_reports_real_sources(tmp_path):
    (tmp_path / "alpha.txt").write_text("Alpha contains the verified launch date.")
    (tmp_path / "beta.md").write_text("Beta discusses an unrelated subject.")
    model = ScriptedResearchModel(("search", "read:doc-0001", "finish"))

    report = research(tmp_path, "What is the launch date?", model=model, max_steps=4)

    assert report.stop_reason == "completed"
    assert report.answer == "Grounded answer [doc-0001]"
    assert tuple(source.relative_path for source in report.sources) == ("alpha.txt",)
    assert tuple(receipt.action for receipt in report.receipts) == (
        "search",
        "read:doc-0001",
        "finish",
    )
    final_prompt = model.requests[-1].messages[0].content
    assert "Alpha contains the verified launch date." in final_prompt
    assert "Beta discusses an unrelated subject." not in final_prompt
    chooser_after_read = model.requests[2].messages[0].content
    assert "Alpha contains the verified launch date." in chooser_after_read


def test_research_budget_stops_repeated_search_without_hidden_extra_actions(tmp_path):
    (tmp_path / "notes.txt").write_text("Evidence")
    model = ScriptedResearchModel(("search", "search", "finish"))

    report = research(tmp_path, "Question", model=model, max_steps=2)

    assert report.stop_reason == "budget_exhausted"
    assert report.answer is None
    assert tuple(receipt.action for receipt in report.receipts) == ("search", "search")
    assert len(model.requests) == 2


def test_model_cannot_choose_an_arbitrary_path_outside_supplied_options(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "public.txt").write_text("Public evidence")
    secret = tmp_path / "secret.txt"
    secret.write_text("TOP SECRET OUTSIDE ROOT")
    model = ScriptedResearchModel(("read:../secret.txt",))

    with pytest.raises(text_ops.InvalidModelOutput, match="configured options"):
        research(docs, "Read the secret", model=model, max_steps=1)

    prompts = "\n".join(
        message.content
        for request in model.requests
        for message in request.messages
        if isinstance(message.content, str)
    )
    assert "TOP SECRET OUTSIDE ROOT" not in prompts


def test_research_rejects_empty_or_unsupported_document_directory(tmp_path):
    (tmp_path / "binary.bin").write_bytes(b"\x00\x01")
    with pytest.raises(ValueError, match="supported documents"):
        research(tmp_path, "Question", model=ScriptedResearchModel(()))


def test_research_skips_hidden_files_and_symlinks(tmp_path):
    (tmp_path / "public.txt").write_text("Public")
    (tmp_path / ".hidden.md").write_text("Hidden")
    outside = tmp_path.parent / "outside-research.txt"
    outside.write_text("Outside")
    (tmp_path / "linked.txt").symlink_to(outside)
    model = ScriptedResearchModel(("finish",))

    report = research(tmp_path, "Question", model=model, max_steps=1)

    assert report.stop_reason == "completed"
    manifest = model.requests[0].messages[0].content
    assert "public.txt" in manifest
    assert ".hidden.md" not in manifest
    assert "linked.txt" not in manifest
    assert "Outside" not in manifest


def test_research_rejects_nonpositive_resource_bounds(tmp_path):
    (tmp_path / "public.txt").write_text("Public")
    model = ScriptedResearchModel(())

    with pytest.raises(ValueError, match="max_documents"):
        research(tmp_path, "Question", model=model, max_documents=0)
    with pytest.raises(ValueError, match="max_chars"):
        research(tmp_path, "Question", model=model, max_chars=0)


@pytest.mark.parametrize(
    "answer, message",
    (
        ("Answer without a citation", "at least one read source"),
        ("Invented citation [doc-invented]", "unknown source"),
        ("Unsupported claim [doc-9999]", "unknown source"),
    ),
)
def test_finish_requires_real_citations_to_sources_that_were_read(
    tmp_path, answer, message
):
    (tmp_path / "public.txt").write_text("Public evidence")
    model = ScriptedResearchModel(("read:doc-0001", "finish"), answer=answer)

    with pytest.raises(ValueError, match=message):
        research(tmp_path, "Question", model=model, max_steps=2)
