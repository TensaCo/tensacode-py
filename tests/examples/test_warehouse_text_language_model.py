import sys
from types import SimpleNamespace

import pytest

from examples.robotics.warehouse.text_language_model import (
    MODEL_ID,
    MODEL_REVISION,
    PinnedTextLanguageModel,
)


def test_adapter_is_pinned_local_only_and_reports_its_actual_source(monkeypatch):
    calls = []

    class Loader:
        @classmethod
        def from_pretrained(cls, identity, **kwargs):
            calls.append((identity, kwargs))
            return cls()

        def to(self, device):
            assert device == "cpu"
            return self

        def eval(self):
            return self

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            set_num_threads=lambda count: calls.append(count),
            bfloat16="bf16",
            __version__="fixture",
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=Loader,
            AutoModelForCausalLM=Loader,
        ),
    )
    model = PinnedTextLanguageModel()
    assert calls[0] == 2
    for identity, options in calls[1:]:
        assert identity == MODEL_ID
        assert options["revision"] == MODEL_REVISION
        assert options["local_files_only"] is True
    provenance = model.provenance()
    assert provenance["model_id"] == MODEL_ID
    assert provenance["revision"] == MODEL_REVISION
    assert len(provenance["adapter_source_sha256"]) == 64
    assert provenance["device"] == "cpu"


@pytest.mark.parametrize(
    "options", [{"threads": 0}, {"threads": 3}, {"threads": True}, {"device": "auto"}]
)
def test_unallocated_adapter_settings_fail_before_loading(options):
    with pytest.raises(ValueError):
        PinnedTextLanguageModel(**options)
