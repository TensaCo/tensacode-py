from abc import ABC
from typing import Any, Generic
from langchain.chains.base import Chain
import inflect
from tensacode.base.base_engine import BaseEngine
from tensacode.llm.mixins.supports_choice import SupportsChoiceMixin
from tensacode.llm.mixins.supports_combine import SupportsCombineMixin
from tensacode.llm.mixins.supports_correct import SupportsCorrectMixin
from tensacode.llm.mixins.supports_create import SupportsCreateMixin
from tensacode.llm.mixins.supports_decide import SupportsDecideMixin
from tensacode.llm.mixins.supports_decode import SupportsDecodeMixin
from tensacode.llm.mixins.supports_encode import SupportsEncodeMixin
from tensacode.llm.mixins.supports_modify import SupportsModifyMixin
from tensacode.llm.mixins.supports_predict import SupportsPredictMixin
from tensacode.llm.mixins.supports_query import SupportsQueryMixin
from tensacode.llm.mixins.supports_retrieve import SupportsRetrieveMixin
from tensacode.llm.mixins.supports_run import SupportsRunMixin
from tensacode.llm.mixins.supports_semantic_transfer import (
    SupportsSemanticTransferMixin,
)
from tensacode.llm.mixins.supports_similarity import SupportsSimilarityMixin
from tensacode.llm.mixins.supports_split import SupportsSplitMixin
from tensacode.llm.mixins.supports_store import SupportsStoreMixin
from tensacode.llm.mixins.supports_style_transfer import (
    SupportsStyleTransferMixin,
)
from tensacode.utils.types import T, R


class LLMEngine(
    Generic[T, R],
    SupportsChoiceMixin[T, R],
    SupportsCombineMixin[T, R],
    SupportsCorrectMixin[T, R],
    SupportsCreateMixin[T, R],
    SupportsDecideMixin[T, R],
    SupportsDecodeMixin[T, R],
    SupportsEncodeMixin[T, R],
    SupportsModifyMixin[T, R],
    SupportsPredictMixin[T, R],
    SupportsQueryMixin[T, R],
    SupportsRetrieveMixin[T, R],
    SupportsRunMixin[T, R],
    SupportsSemanticTransferMixin[T, R],
    SupportsSimilarityMixin[T, R],
    SupportsSplitMixin[T, R],
    SupportsStoreMixin[T, R],
    SupportsStyleTransferMixin[T, R],
    BaseEngine[T, R],
):
    pass
