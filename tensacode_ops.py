
from typing import ClassVar, Dict, Type, Any, Literal, Tuple
from abc import abstractmethod
import pydantic
from .base_engine import BaseEngine

EngineType = Literal["nn", "llm", "gnn"]

class ObjectType:
    generic_name: ClassVar[str]

    @property
    def specific_name(self) -> str:
        return self.generic_name

    def __repr__(self):
        return f"{self.specific_name}"

class TextType(ObjectType):
    generic_name = "text"

class ImageType(ObjectType):
    generic_name = "image"
    height: int
    width: int
    channels: int

    @property
    def specific_name(self) -> str:
        return f"{self.generic_name} ({self.height}x{self.width}x{self.channels})"

class AudioType(ObjectType):
    generic_name = "audio"
    sample_rate: int

    @property
    def specific_name(self) -> str:
        return f"{self.generic_name} ({self.sample_rate}Hz)"

class VideoType(ObjectType):
    generic_name = "video"
    height: int
    width: int
    channels: int

    @property
    def specific_name(self) -> str:
        return f"{self.generic_name} ({self.height}x{self.width}x{self.channels})"

class DictType(ObjectType):
    generic_name = "dict"
    dict_type: Dict[str, ObjectType]

    @property
    def specific_name(self) -> str:
        return f"{self.generic_name} ({self.dict_type})"

class BaseOp(pydantic.BaseModel):
    OpRegistry: ClassVar[Dict[Tuple[EngineType, str], Type['BaseOp']]] = {}

    def __init_subclass__(cls):
        assert cls.engine_type is not None, f"engine_type is not defined on subclass {cls.__name__}"
        assert cls.op_type is not None, f"op_type is not defined on subclass {cls.__name__}"
        cls.OpRegistry[cls.engine_type, cls.op_type] = cls

    engine_type: ClassVar[EngineType]
    op_type: ClassVar[str]

    @abstractmethod
    def execute(self, engine: BaseEngine, input: Any):
        ...

def get_input_type(input: Any) -> str:
    if isinstance(input, str):
        return "text"
    elif isinstance(input, ImageType):
        return "image"
    elif isinstance(input, AudioType):
        return "audio"
    elif isinstance(input, VideoType):
        return "video"
    elif isinstance(input, dict):
        return "dict"
    else:
        return "unknown"

class EncodeOp(BaseOp):
    op_type: ClassVar[str] = "encode"

    @abstractmethod
    def execute(self, engine: BaseEngine, input: Any):
        ...

class DecodeOp(BaseOp):
    op_type: ClassVar[str] = "decode"

    @abstractmethod
    def execute(self, engine: BaseEngine, input: Any):
        ...

class DecideOp(BaseOp):
    op_type: ClassVar[str] = "decide"

    @abstractmethod
    def execute(self, engine: BaseEngine, input: Any):
        ...

class CallOp(BaseOp):
    op_type: ClassVar[str] = "call"

    @abstractmethod
    def execute(self, engine: BaseEngine, input: Any):
        ...

class ChoiceOp(BaseOp):
    op_type: ClassVar[str] = "choice"

    @abstractmethod
    def execute(self, engine: BaseEngine, input: Any):
        ...

class CombineOp(BaseOp):
    op_type: ClassVar[str] = "combine"

    @abstractmethod
    def execute(self, engine: BaseEngine, input: Any):
        ...

class ConvertOp(BaseOp):
    op_type: ClassVar[str] = "convert"

    @abstractmethod
    def execute(self, engine: BaseEngine, input: Any):
        ...

class CorrectOp(BaseOp):
    op_type: ClassVar[str] = "correct"

    @abstractmethod
    def execute(self, engine: BaseEngine, input: Any):
        ...

class ModifyOp(BaseOp):
    op_type: ClassVar[str] = "modify"

    @abstractmethod
    def execute(self, engine: BaseEngine, input: Any):
        ...

class PredictOp(BaseOp):
    op_type: ClassVar[str] = "predict"

    @abstractmethod
    def execute(self, engine: BaseEngine, input: Any):
        ...

class ProgramOp(BaseOp):
    op_type: ClassVar[str] = "program"

    @abstractmethod
    def execute(self, engine: BaseEngine, input: Any):
        ...

class QueryOp(BaseOp):
    op_type: ClassVar[str] = "query"

    @abstractmethod
    def execute(self, engine: BaseEngine, input: Any):
        ...

class RetrieveOp(BaseOp):
    op_type: ClassVar[str] = "retrieve"

    @abstractmethod
    def execute(self, engine: BaseEngine, input: Any):
        ...

class RunOp(BaseOp):
    op_type: ClassVar[str] = "run"

    @abstractmethod
    def execute(self, engine: BaseEngine, input: Any):
        ...

class SemanticTransferOp(BaseOp):
    op_type: ClassVar[str] = "semantictransfer"

    @abstractmethod
    def execute(self, engine: BaseEngine, input: Any):
        ...

class SimilarityOp(BaseOp):
    op_type: ClassVar[str] = "similarity"

    @abstractmethod
    def execute(self, engine: BaseEngine, input: Any):
        ...

class SplitOp(BaseOp):
    op_type: ClassVar[str] = "split"

    @abstractmethod
    def execute(self, engine: BaseEngine, input: Any):
        ...

class StoreOp(BaseOp):
    op_type: ClassVar[str] = "store"

    @abstractmethod
    def execute(self, engine: BaseEngine, input: Any):
        ...

class StyleTransferOp(BaseOp):
    op_type: ClassVar[str] = "styletransfer"

    @abstractmethod
    def execute(self, engine: BaseEngine, input: Any):
        ...

class NNBaseOp(BaseOp):
    engine_type: ClassVar[EngineType] = "nn"

class NNEncodeTextOp(NNBaseOp, EncodeOp):
    op_type: ClassVar[str] = "encode_text"

    def execute(self, engine: BaseEngine, input: Any):
        ...

class NNEncodeImageOp(NNBaseOp, EncodeOp):
    op_type: ClassVar[str] = "encode_image"

    def execute(self, engine: BaseEngine, input: Any):
        ...

class NNEncodeAudioOp(NNBaseOp, EncodeOp):
    op_type: ClassVar[str] = "encode_audio"

    def execute(self, engine: BaseEngine, input: Any):
        ...

class NNEncodeVideoOp(NNBaseOp, EncodeOp):
    op_type: ClassVar[str] = "encode_video"

    def execute(self, engine: BaseEngine, input: Any):
        ...

class NNEncodeDictOp(NNBaseOp, EncodeOp):
    op_type: ClassVar[str] = "encode_dict"

    def execute(self, engine: BaseEngine, input: Any):
        ...

class NNDecodeTextOp(NNBaseOp, DecodeOp):
    op_type: ClassVar[str] = "decode_text"

    def execute(self, engine: BaseEngine, input: Any):
        ...

class NNDecodeImageOp(NNBaseOp, DecodeOp):
    op_type: ClassVar[str] = "decode_image"

    def execute(self, engine: BaseEngine, input: Any):
        ...

class NNDecodeAudioOp(NNBaseOp, DecodeOp):
    op_type: ClassVar[str] = "decode_audio"

    def execute(self, engine: BaseEngine, input: Any):
        ...

class NNDecodeVideoOp(NNBaseOp, DecodeOp):
    op_type: ClassVar[str] = "decode_video"

    def execute(self, engine: BaseEngine, input: Any):
        ...

class NNDecodeDictOp(NNBaseOp, DecodeOp):
    op_type: ClassVar[str] = "decode_dict"

    def execute(self, engine: BaseEngine, input: Any):
        ...

class NNDecideOp(NNBaseOp, DecideOp):
    def execute(self, engine: BaseEngine, input: Any):
        ...

class NNCallOp(NNBaseOp, CallOp):
    def execute(self, engine: BaseEngine, input: Any):
        ...

class NNChoiceOp(NNBaseOp, ChoiceOp):
    def execute(self, engine: BaseEngine, input: Any):
        ...

class NNCombineOp(NNBaseOp, CombineOp):
    def execute(self, engine: BaseEngine, input: Any):
        ...

class NNConvertOp(NNBaseOp, ConvertOp):
    def execute(self, engine: BaseEngine, input: Any):
        ...

class NNCorrectOp(NNBaseOp, CorrectOp):
    def execute(self, engine: BaseEngine, input: Any):
        ...

class NNModifyOp(NNBaseOp, ModifyOp):
    def execute(self, engine: BaseEngine, input: Any):
        ...

class NNPredictOp(NNBaseOp, PredictOp):
    def execute(self, engine: BaseEngine, input: Any):
        ...

class NNProgramOp(NNBaseOp, ProgramOp):
    def execute(self, engine: BaseEngine, input: Any):
        ...

class NNQueryOp(NNBaseOp, QueryOp):
    def execute(self, engine: BaseEngine, input: Any):
        ...

class NNRetrieveOp(NNBaseOp, RetrieveOp):
    def execute(self, engine: BaseEngine, input: Any):
        ...

class NNRunOp(NNBaseOp, RunOp):
    def execute(self, engine: BaseEngine, input: Any):
        ...

class NNSemanticTransferOp(NNBaseOp, SemanticTransferOp):
    def execute(self, engine: BaseEngine, input: Any):
        ...

class NNSimilarityOp(NNBaseOp, SimilarityOp):
    def execute(self, engine: BaseEngine, input: Any):
        ...

class NNSplitOp(NNBaseOp, SplitOp):
    def execute(self, engine: BaseEngine, input: Any):
        ...

class NNStoreOp(NNBaseOp, StoreOp):
    def execute(self, engine: BaseEngine, input: Any):
        ...

class NNStyleTransferOp(NNBaseOp, StyleTransferOp):
    def execute(self, engine: BaseEngine, input: Any):
        ...

class LLMBaseOp(BaseOp):
    engine_type: ClassVar[EngineType] = "llm"

class LLMEncodeTextOp(LLMBaseOp, EncodeOp):
    op_type: ClassVar[str] = "encode_text"

    def execute(self, engine: BaseEngine, input: Any):
        ...

class LLMEncodeImageOp(LLMBaseOp, EncodeOp):
    op_type: ClassVar[str] = "encode_image"

    def execute(self, engine: BaseEngine, input: Any):
        ...

class LLMEncodeAudioOp(LLMBaseOp, EncodeOp):
    op_type: ClassVar[str] = "encode_audio"

    def execute(self, engine: BaseEngine, input: Any):
        ...

class LLMEncodeVideoOp(LLMBaseOp, EncodeOp):
    op_type: ClassVar[str] = "encode_video"

    def execute(self, engine: BaseEngine, input: Any):
        ...

class LLMEncodeDictOp(LLMBaseOp, EncodeOp):
    op_type: ClassVar[str] = "encode_dict"

    def execute(self, engine: BaseEngine, input: Any):
        ...

class LLMDecodeTextOp(LLMBaseOp, DecodeOp):
    op_type: ClassVar[str] = "decode_text"

    def execute(self, engine: BaseEngine, input: Any):
        ...

class LLMDecodeImageOp(LLMBaseOp, DecodeOp):
    op_type: ClassVar[str] = "decode_image"

    def execute(self, engine: BaseEngine, input: Any):
        ...

class LLMDecodeAudioOp(LLMBaseOp, DecodeOp):
    op_type: ClassVar[str] = "decode_audio"

    def execute(self, engine: BaseEngine, input: Any):
        ...

class LLMDecodeVideoOp(LLMBaseOp, DecodeOp):
    op_type: ClassVar[str] = "decode_video"

    def execute(self, engine: BaseEngine, input: Any):
        ...

class LLMDecodeDictOp(LLMBaseOp, DecodeOp):
    op_type: ClassVar[str] = "decode_dict"

    def execute(self, engine: BaseEngine, input: Any):
        ...

class LLMDecideOp(LLMBaseOp, DecideOp):
    def execute(self, engine: BaseEngine, input: Any):
        ...

class LLMCallOp(LLMBaseOp, CallOp):
    def execute(self, engine: BaseEngine, input: Any):
        ...

class LLMChoiceOp(LLMBaseOp, ChoiceOp):
    def execute(self, engine: BaseEngine, input: Any):
        ...

class LLMCombineOp(LLMBaseOp, CombineOp):
    def execute(self, engine: BaseEngine, input: Any):
        ...

class LLMConvertOp(LLMBaseOp, ConvertOp):
    def execute(self, engine: BaseEngine, input: Any):
        ...

class LLMCorrectOp(LLMBaseOp, CorrectOp):
    def execute(self, engine: BaseEngine, input: Any):
        ...

class LLMModifyOp(LLMBaseOp, ModifyOp):
    def execute(self, engine: BaseEngine, input: Any):
        ...

class LLMPredictOp(LLMBaseOp, PredictOp):
    def execute(self, engine: BaseEngine, input: Any):
        ...

class LLMProgramOp(LLMBaseOp, ProgramOp):
    def execute(self, engine: BaseEngine, input: Any):
        ...

class LLMQueryOp(LLMBaseOp, QueryOp):
    def execute(self, engine: BaseEngine, input: Any):
        ...

class LLMRetrieveOp(LLMBaseOp, RetrieveOp):
    def execute(self, engine: BaseEngine, input: Any):
        ...

class LLMRunOp(LLMBaseOp, RunOp):
    def execute(self, engine: BaseEngine, input: Any):
        ...

class LLMSemanticTransferOp(LLMBaseOp, SemanticTransferOp):
    def execute(self, engine: BaseEngine, input: Any):
        ...

class LLMSimilarityOp(LLMBaseOp, SimilarityOp):
    def execute(self, engine: BaseEngine, input: Any):
        ...

class LLMSplitOp(LLMBaseOp, SplitOp):
    def execute(self, engine: BaseEngine, input: Any):
        ...

class LLMStoreOp(LLMBaseOp, StoreOp):
    def execute(self, engine: BaseEngine, input: Any):
        ...

class LLMStyleTransferOp(LLMBaseOp, StyleTransferOp):
    def execute(self, engine: BaseEngine, input: Any):
        ...

class GNNBaseOp(BaseOp):
    engine_type: ClassVar[EngineType] = "gnn"

class GNNEncodeTextOp(GNNBaseOp, EncodeOp):
    op_type: ClassVar[str] = "encode_text"

    def execute(self, engine: BaseEngine, input: Any):
        ...

class GNNEncodeImageOp(GNNBaseOp, EncodeOp):
    op_type: ClassVar[str] = "encode_image"

    def execute(self, engine: BaseEngine, input: Any):
        ...

class GNNEncodeAudioOp(GNNBaseOp, EncodeOp):
    op_type: ClassVar[str] = "encode_audio"

    def execute(self, engine: BaseEngine, input: Any):
        ...

class GNNEncodeVideoOp(GNNBaseOp, EncodeOp):
    op_type: ClassVar[str] = "encode_video"

    def execute(self, engine: BaseEngine, input: Any):
        ...

class GNNEncodeDictOp(GNNBaseOp, EncodeOp):
    op_type: ClassVar[str] = "encode_dict"

    def execute(self, engine: BaseEngine, input: Any):
        ...

class GNNDecodeTextOp(GNNBaseOp, DecodeOp):
    op_type: ClassVar[str] = "decode_text"

    def execute(self, engine: BaseEngine, input: Any):
        ...

class GNNDecodeImageOp(GNNBaseOp, DecodeOp):
    op_type: ClassVar[str] = "decode_image"

    def execute(self, engine: BaseEngine, input: Any):
        ...

class GNNDecodeAudioOp(GNNBaseOp, DecodeOp):
    op_type: ClassVar[str] = "decode_audio"

    def execute(self, engine: BaseEngine, input: Any):
        ...

class GNNDecodeVideoOp(GNNBaseOp, DecodeOp):
    op_type: ClassVar[str] = "decode_video"

    def execute(self, engine: BaseEngine, input: Any):
        ...

class GNNDecodeDictOp(GNNBaseOp, DecodeOp):
    op_type: ClassVar[str] = "decode_dict"

    def execute(self, engine: BaseEngine, input: Any):
        ...

class GNNDecideOp(GNNBaseOp, DecideOp):
    def execute(self, engine: BaseEngine, input: Any):
        ...

class GNNCallOp(GNNBaseOp, CallOp):
    def execute(self, engine: BaseEngine, input: Any):
        ...

class GNNChoiceOp(GNNBaseOp, ChoiceOp):
    def execute(self, engine: BaseEngine, input: Any):
        ...

class GNNCombineOp(GNNBaseOp, CombineOp):
    def execute(self, engine: BaseEngine, input: Any):
        ...

class GNNConvertOp(GNNBaseOp, ConvertOp):
    def execute(self, engine: BaseEngine, input: Any):
        ...

class GNNCorrectOp(GNNBaseOp, CorrectOp):
    def execute(self, engine: BaseEngine, input: Any):
        ...

class GNNModifyOp(GNNBaseOp, ModifyOp):
    def execute(self, engine: BaseEngine, input: Any):
        ...

class GNNPredictOp(GNNBaseOp, PredictOp):
    def execute(self, engine: BaseEngine, input: Any):
        ...

class GNNProgramOp(GNNBaseOp, ProgramOp):
    def execute(self, engine: BaseEngine, input: Any):
        ...

class GNNQueryOp(GNNBaseOp, QueryOp):
    def execute(self, engine: BaseEngine, input: Any):
        ...

class GNNRetrieveOp(GNNBaseOp, RetrieveOp):
    def execute(self, engine: BaseEngine, input: Any):
        ...

class GNNRunOp(GNNBaseOp, RunOp):
    def execute(self, engine: BaseEngine, input: Any):
        ...

class GNNSemanticTransferOp(GNNBaseOp, SemanticTransferOp):
    def execute(self, engine: BaseEngine, input: Any):
        ...

class GNNSimilarityOp(GNNBaseOp, SimilarityOp):
    def execute(self, engine: BaseEngine, input: Any):
        ...

class GNNSplitOp(GNNBaseOp, SplitOp):
    def execute(self, engine: BaseEngine, input: Any):
        ...

class GNNStoreOp(GNNBaseOp, StoreOp):
    def execute(self, engine: BaseEngine, input: Any):
        ...

class GNNStyleTransferOp(GNNBaseOp, StyleTransferOp):
    def execute(self, engine: BaseEngine, input: Any):
        ...
