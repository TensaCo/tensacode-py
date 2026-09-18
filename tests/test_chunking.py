"""Automatization: compiling a repeated path, and giving it up when the world changes."""

from __future__ import annotations

from tensacode.chunking import Chunk, Chunks, Trace, divergent, mark

PATH = Trace("count", taken=(0, 2, 3), skipped=(1,))
OTHER = Trace("count", taken=(0, 1, 3), skipped=(2,))


def test_a_paths_identity_is_the_path_not_the_object():
    assert PATH.shape == Trace("count", (0, 2, 3), (1,)).shape
    assert PATH.shape != OTHER.shape
    assert PATH.shape != Trace("read", (0, 2, 3), (1,)).shape


def test_what_went_wrong_is_part_of_the_path():
    """A run where a step errored is a different path from one where it did not."""
    troubled = Trace("count", (0, 2, 3), (1,), marks=((2, "errors"),))
    assert troubled.shape != PATH.shape


def test_a_chunk_compiles_only_after_the_same_path_repeats():
    chunks = Chunks(repeats=3)
    assert chunks.record(PATH) is None
    assert chunks.record(PATH) is None
    chunk = chunks.record(PATH)
    assert isinstance(chunk, Chunk) and chunk.from_runs == 3
    assert chunks.chunk_for("count") is chunk
    assert chunk.assumes_skipped(1) and not chunk.assumes_skipped(2)
    assert "2 steps" not in chunk.describe() and "3 steps" in chunk.describe()


def test_a_different_path_does_not_count_towards_a_chunk():
    chunks = Chunks(repeats=3)
    chunks.record(PATH), chunks.record(OTHER), chunks.record(PATH)
    assert chunks.chunk_for("count") is None


def test_a_run_that_did_not_finish_teaches_nothing():
    chunks = Chunks(repeats=2)
    chunks.record(PATH)
    chunks.record(Trace("count", (0, 2), (1,), status="failed"))
    chunks.record(PATH)
    assert chunks.chunk_for("count") is None, "a failed run must not be half of a compiled skill"


def test_a_retired_chunk_is_not_used_and_the_expanded_form_survives():
    chunks = Chunks(repeats=1)
    chunk = chunks.record(PATH)
    chunks.used(chunk)
    chunks.retire(chunk, "errors, which the recorded runs never had")
    assert chunks.chunk_for("count") is None
    assert chunks.compiled["count"] is chunk and chunk.retired  # kept, so it can be explained
    assert chunk.fallbacks == 1 and chunks.runs["count"] == []
    assert chunks.stats() == {"compiled": 0, "retired": 1, "uses": 1, "fallbacks": 1,
                              "procedures": {"count": chunk.describe()}}
    assert "retired" in chunk.describe()


def test_chunking_can_be_switched_off_without_losing_what_was_learned():
    chunks = Chunks(repeats=1)
    chunks.record(PATH)
    chunks.enabled = False
    assert chunks.chunk_for("count") is None
    chunks.enabled = True
    assert chunks.chunk_for("count") is not None


def test_a_good_output_is_not_trouble():
    """The assistant always fills in error_summary, so reading that as trouble would retire
    every chunk on first use."""
    assert mark({"out": "hello", "ok": True, "errors": [], "error_summary": "hello", "timed_out": False}) == ""


def test_trouble_is_found_under_a_steps_own_name():
    bindings = {"sib_ok": False, "sib_errors": ["cannot statx"], "sib_error_summary": "cannot statx"}
    assert mark(bindings) == "sib_errors"


def test_trouble_marks_that_are_flags_or_text():
    assert mark({"timed_out": True}) == "timed_out"
    assert mark({"problem": "the click was rejected"}) == "problem"
    assert mark({"unavailable": ""}) == ""
    assert mark({"ok": False}) == "ok"


def test_divergence_is_a_difference_in_both_directions():
    troubled = {"ok": False, "errors": ["No such file"]}
    fine = {"ok": True, "errors": []}
    assert divergent(fine, expected="") == ""
    assert divergent(troubled, expected="errors") == "", "a step that always errors is normal for it"
    assert "never had" in divergent(troubled, expected="")
    assert "every recorded run had" in divergent(fine, expected="errors")
    assert "where the recorded runs had" in divergent({"timed_out": True}, expected="errors")
