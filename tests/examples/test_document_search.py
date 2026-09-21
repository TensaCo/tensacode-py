import importlib.util
from pathlib import Path

import pytest

from tensorcode.ops import llm


def example():
    path = Path(__file__).parents[2] / "examples/document_search.py"
    spec = importlib.util.spec_from_file_location("document_search_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ScriptedModel:
    def __init__(self, *outputs):
        self.outputs = list(outputs)
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        return self.outputs.pop(0)


def test_loads_visible_utf8_documents_in_deterministic_order_and_skips_symlinks(tmp_path):
    mod = example()
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "z.md").write_text("Zebra reference", encoding="utf-8")
    (docs / "a.txt").write_text("Alpha reference", encoding="utf-8")
    (docs / "ignore.csv").write_text("not searched", encoding="utf-8")
    hidden = docs / ".hidden"
    hidden.mkdir()
    (hidden / "secret.md").write_text("secret", encoding="utf-8")
    nested = docs / "nested"
    nested.mkdir()
    (nested / "b.md").write_text("Beta reference", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    (docs / "linked.txt").symlink_to(outside)
    (docs / "linked-dir").symlink_to(nested, target_is_directory=True)

    chunks = mod.load_chunks(
        docs, max_files=10, max_bytes_per_file=100, chunk_chars=100, max_chunks=10
    )

    assert [chunk.source_id for chunk in chunks] == [
        "a.txt#chunk-0001",
        "nested/b.md#chunk-0001",
        "z.md#chunk-0001",
    ]
    assert [chunk.text for chunk in chunks] == [
        "Alpha reference",
        "Beta reference",
        "Zebra reference",
    ]


def test_retrieves_explicit_chunks_then_answers_with_validated_citations():
    mod = example()
    chunks = (
        mod.Chunk("guide.md#chunk-0001", "guide.md", 0, 20, "Reset from Settings."),
        mod.Chunk("policy.txt#chunk-0001", "policy.txt", 0, 17, "Refunds take 5 days."),
    )
    model = ScriptedModel(
        llm.ModelOutput(
            structured={
                "keys": ["guide.md#chunk-0001"],
                "scores": None,
                "abstained": False,
            }
        ),
        llm.ModelOutput(text="Open Settings to reset it [guide.md#chunk-0001]."),
    )

    result = mod.search_documents(
        chunks,
        query="How do I reset it?",
        model=model,
        top_k=1,
        max_context_chars=100,
    )

    assert result["answer"].endswith("[guide.md#chunk-0001].")
    assert result["abstained"] is False
    assert result["sources"] == [
        {
            "id": "guide.md#chunk-0001",
            "path": "guide.md",
            "start": 0,
            "end": 20,
            "excerpt": "Reset from Settings.",
        }
    ]
    assert model.requests[0].schema_name == "tensorcode.retrieve"
    candidate_schema = model.requests[0].response_schema["properties"]["keys"]["items"]
    assert candidate_schema["enum"] == ["guide.md#chunk-0001", "policy.txt#chunk-0001"]
    assert "Refunds take 5 days." in candidate_schema["description"]
    answer_prompt = model.requests[1].messages[-1].content
    assert "guide.md#chunk-0001" in answer_prompt
    assert "Reset from Settings." in answer_prompt
    assert "Refunds take 5 days." not in answer_prompt


def test_abstention_does_not_call_answer_model():
    mod = example()
    chunk = mod.Chunk("a.txt#chunk-0001", "a.txt", 0, 4, "text")
    model = ScriptedModel(
        llm.ModelOutput(structured={"keys": [], "scores": None, "abstained": True})
    )
    result = mod.search_documents(
        (chunk,), query="unknown", model=model, top_k=1, max_context_chars=100
    )
    assert result == {
        "query": "unknown",
        "answer": None,
        "abstained": True,
        "scores": None,
        "sources": [],
    }
    assert len(model.requests) == 1


def test_rejects_missing_or_unretrieved_answer_citations():
    mod = example()
    chunk = mod.Chunk("a.txt#chunk-0001", "a.txt", 0, 4, "text")
    for answer in ("No citation.", "Wrong [other.txt#chunk-0001]."):
        model = ScriptedModel(
            llm.ModelOutput(
                structured={
                    "keys": ["a.txt#chunk-0001"],
                    "scores": None,
                    "abstained": False,
                }
            ),
            llm.ModelOutput(text=answer),
        )
        with pytest.raises(ValueError, match="citation"):
            mod.search_documents(
                (chunk,), query="question", model=model, top_k=1, max_context_chars=100
            )


def test_rejects_oversized_candidate_context_before_any_model_call():
    mod = example()
    chunks = (
        mod.Chunk("a#chunk-0001", "a.txt", 0, 6, "abcdef"),
        mod.Chunk("b#chunk-0001", "b.txt", 0, 6, "ghijkl"),
    )
    model = ScriptedModel()
    with pytest.raises(ValueError, match="max_candidate_chars"):
        mod.search_documents(
            chunks,
            query="question",
            model=model,
            top_k=1,
            max_context_chars=100,
            max_candidate_chars=10,
        )
    assert model.requests == []


def test_rejects_oversized_files_and_chunk_limits(tmp_path):
    mod = example()
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "large.txt").write_text("1234567890", encoding="utf-8")
    with pytest.raises(ValueError, match="max_bytes_per_file"):
        mod.load_chunks(
            docs, max_files=2, max_bytes_per_file=5, chunk_chars=3, max_chunks=10
        )
    with pytest.raises(ValueError, match="max_chunks"):
        mod.load_chunks(
            docs, max_files=2, max_bytes_per_file=20, chunk_chars=3, max_chunks=2
        )
