class AnyValue:
    generic_name: ClassVar[str]
    type: ClassVar[Type[Any]]
    data: Any

    @abstractmethod
    def text_description(self, config: EngineConfig) -> str:
        pass

    def isinstance(self, other: Any) -> bool:
        return isinstance(other, self.type)


class PrimitiveValue(AnyValue):
    generic_name = "primitive"

    def text_description(self, config: EngineConfig) -> str:
        match config.render_format:
            case "english":
                return f"Primitive type"
            case "chinese":
                return f"基本类型"
            case "python":
                return f"primitive"
            case "typescript":
                return f"Primitive"
            case _:
                raise NotImplementedError(
                    f"Invalid render_format: {config.render_format}"
                )


class IntegerValue(PrimitiveValue):
    type = int
    generic_name = "integer"

    def text_description(self, config: EngineConfig) -> str:
        match config.render_format:
            case "english":
                return f"Integer type"
            case "chinese":
                return f"整数类型"
            case "python":
                return f"int"
            case "typescript":
                return f"number"
            case _:
                raise NotImplementedError(
                    f"Invalid render_format: {config.render_format}"
                )


class FloatValue(PrimitiveValue):
    type = float
    generic_name = "float"

    def text_description(self, config: EngineConfig) -> str:
        match config.render_format:
            case "english":
                return f"Floating-point number"
            case "chinese":
                return f"浮点数"
            case "python":
                return f"float"
            case "typescript":
                return f"number"
            case _:
                raise NotImplementedError(
                    f"Invalid render_format: {config.render_format}"
                )


class BooleanValue(PrimitiveValue):
    type = bool
    generic_name = "boolean"

    def text_description(self, config: EngineConfig) -> str:
        match config.render_format:
            case "english":
                return f"Boolean type"
            case "chinese":
                return f"布尔类型"
            case "python":
                return f"bool"
            case "typescript":
                return f"boolean"
            case _:
                raise NotImplementedError(
                    f"Invalid render_format: {config.render_format}"
                )


class ComplexNumberValue(PrimitiveValue):
    type = complex
    generic_name = "complex_number"

    def text_description(self, config: EngineConfig) -> str:
        match config.render_format:
            case "english":
                return f"Complex number type"
            case "chinese":
                return f"复数类型"
            case "python":
                return f"complex"
            case "typescript":
                return f"Complex"
            case _:
                raise NotImplementedError(
                    f"Invalid render_format: {config.render_format}"
                )


class StringValue(PrimitiveValue):
    type = str
    generic_name = "string"

    def text_description(self, config: EngineConfig) -> str:
        match config.render_format:
            case "english":
                return f"String type"
            case "chinese":
                return f"字符串类型"
            case "python":
                return f"str"
            case "typescript":
                return f"string"
            case _:
                raise NotImplementedError(
                    f"Invalid render_format: {config.render_format}"
                )


class ContainerValue(AnyValue):
    generic_name = "container"


class ListValue(ContainerValue):
    type = list
    generic_name = "list"
    element_type: AnyValue

    def text_description(self, config: EngineConfig) -> str:
        match config.render_format:
            case "english":
                return f"List of {self.element_type}"
            case "chinese":
                return f"列表包含 {self.element_type}"
            case "python":
                return f"list({self.element_type})"
            case "typescript":
                return f"Array<{self.element_type}>"
            case _:
                raise NotImplementedError(
                    f"Invalid render_format: {config.render_format}"
                )


class TupleValue(ContainerValue):
    type = tuple
    generic_name = "tuple"
    element_types: List[AnyValue]

    def text_description(self, config: EngineConfig) -> str:
        match config.render_format:
            case "english":
                return f"Tuple of {', '.join(str(t) for t in self.element_types)}"
            case "chinese":
                return f"元组包含 {', '.join(str(t) for t in self.element_types)}"
            case "python":
                return f"tuple({', '.join(str(t) for t in self.element_types)})"
            case "typescript":
                return f"[{', '.join(str(t) for t in self.element_types)}]"
            case _:
                raise NotImplementedError(
                    f"Invalid render_format: {config.render_format}"
                )


class DictValue(ContainerValue):
    type = dict
    generic_name = "dict"
    key_type: AnyValue
    value_type: AnyValue

    def text_description(self, config: EngineConfig) -> str:
        match config.render_format:
            case "english":
                return f"Dictionary with keys of {self.key_type} and values of {self.value_type}"
            case "chinese":
                return f"字典，键为 {self.key_type}，值为 {self.value_type}"
            case "python":
                return f"dict[{self.key_type}: {self.value_type}]"
            case "typescript":
                return f"{{[key: {self.key_type}]: {self.value_type}}}"
            case _:
                raise NotImplementedError(
                    f"Invalid render_format: {config.render_format}"
                )


class SetValue(ContainerValue):
    type = set
    generic_name = "set"
    element_type: AnyValue

    def text_description(self, config: EngineConfig) -> str:
        match config.render_format:
            case "english":
                return f"Set of {self.element_type}"
            case "chinese":
                return f"集合包含 {self.element_type}"
            case "python":
                return f"set({self.element_type})"
            case "typescript":
                return f"Set<{self.element_type}>"
            case _:
                raise NotImplementedError(
                    f"Invalid render_format: {config.render_format}"
                )


class FrozenSetValue(SetValue):
    type = frozenset
    generic_name = "frozenset"

    def text_description(self, config: EngineConfig) -> str:
        match config.render_format:
            case "english":
                return f"Frozen set of {self.element_type}"
            case "chinese":
                return f"冻结集合包含 {self.element_type}"
            case "python":
                return f"frozenset({self.element_type})"
            case "typescript":
                return f"FrozenSet<{self.element_type}>"
            case _:
                raise NotImplementedError(
                    f"Invalid render_format: {config.render_format}"
                )


class ModuleValue(AnyValue):
    generic_name = "module"

    def text_description(self, config: EngineConfig) -> str:
        match config.render_format:
            case "english":
                return f"Module {self.module_name}"
            case "chinese":
                return f"模块 {self.module_name}"
            case "python":
                return f"module {self.module_name}"
            case "typescript":
                return f"Module<{self.module_name}>"
            case _:
                raise NotImplementedError(
                    f"Invalid render_format: {config.render_format}"
                )


class ClassValue(AnyValue):
    generic_name = "class"
    class_name: str

    def text_description(self, config: EngineConfig) -> str:
        match config.render_format:
            case "english":
                return f"Class {self.class_name}"
            case "chinese":
                return f"类 {self.class_name}"
            case "python":
                return f"class {self.class_name}"
            case "typescript":
                return f"Class<{self.class_name}>"
            case _:
                raise NotImplementedError(
                    f"Invalid render_format: {config.render_format}"
                )


class InstanceValue(AnyValue):
    generic_name = "instance"
    class_value: ClassValue

    def text_description(self, config: EngineConfig) -> str:
        match config.render_format:
            case "english":
                return f"Instance of {self.class_value}"
            case "chinese":
                return f"实例属于 {self.class_value}"
            case "python":
                return f"instance of {self.class_value}"
            case "typescript":
                return f"Instance<{self.class_value}>"
            case _:
                raise NotImplementedError(
                    f"Invalid render_format: {config.render_format}"
                )


class IteratorValue(AnyValue):
    generic_name = "iterator"
    element_value: AnyValue

    def text_description(self, config: EngineConfig) -> str:
        match config.render_format:
            case "english":
                return f"Iterator of {self.element_value}"
            case "chinese":
                return f"迭代器包含 {self.element_value}"
            case "python":
                return f"iterator({self.element_value})"
            case "typescript":
                return f"Iterator<{self.element_value}>"
            case _:
                raise NotImplementedError(
                    f"Invalid render_format: {config.render_format}"
                )


class StreamValue(AnyValue):
    generic_name = "stream"
    element_type: type[AnyValue]

    def text_description(self, config: EngineConfig) -> str:
        match config.render_format:
            case "english":
                return f"Stream of {self.element_type}"
            case "chinese":
                return f"流包含 {self.element_type}"
            case "python":
                return f"stream({self.element_type})"
            case "typescript":
                return f"Stream<{self.element_type}>"
            case _:
                raise NotImplementedError(
                    f"Invalid render_format: {config.render_format}"
                )


class FunctionValue(AnyValue):
    type = types.FunctionType
    generic_name = "function"
    arg_types: List[AnyValue]
    return_type: AnyValue

    def text_description(self, config: EngineConfig) -> str:
        args = ", ".join(str(t) for t in self.arg_types)
        match config.render_format:
            case "english":
                return f"Function that takes in {args} and returns {self.return_type}"
            case "chinese":
                return f"函数，输入 {args}，返回 {self.return_type}"
            case "python":
                return f"Function[[{args}], {self.return_type}]"
            case "typescript":
                return f"({args}) -> {self.return_type}"
            case _:
                raise NotImplementedError(
                    f"Invalid render_format: {config.render_format}"
                )


class CoRoutineValue(FunctionValue):
    generic_name = "generator"
    yield_type: AnyValue
    send_type: AnyValue

    def text_description(self, config: EngineConfig) -> str:
        match config.render_format:
            case "english":
                return f"Generator yielding {self.yield_type}, sending {self.send_type}, returning {self.return_type}"
            case "chinese":
                return f"生成器产出 {self.yield_type}，发送 {self.send_type}，返回 {self.return_type}"
            case "python":
                return f"generator(yielding={self.yield_type}, sending={self.send_type}, returning={self.return_type})"
            case "typescript":
                return f"Generator<{self.yield_type}, {self.send_type}, {self.return_type}>"
            case _:
                raise NotImplementedError(
                    f"Invalid render_format: {config.render_format}"
                )


class LambdaValue(FunctionType):
    type = types.LambdaType
    generic_name = "lambda"

    def text_description(self, config: EngineConfig) -> str:
        args = ", ".join(str(t) for t in self.arg_types)
        match config.render_format:
            case "english":
                return f"Lambda function that takes in {args} and returns {self.return_type}"
            case "chinese":
                return f"匿名函数，输入 {args}，返回 {self.return_type}"
            case "python":
                return f"lambda([{args}], {self.return_type})"
            case "typescript":
                return f"({args}) -> {self.return_type}"
            case _:
                raise NotImplementedError(
                    f"Invalid render_format: {config.render_format}"
                )


class CodeValue(AnyValue):
    type = types.CodeType
    generic_name = "code"

    def text_description(self, config: EngineConfig) -> str:
        match config.render_format:
            case "english":
                return "Code object"
            case "chinese":
                return "代码对象"
            case "python":
                return "code"
            case "typescript":
                return "Code"
            case _:
                raise NotImplementedError(
                    f"Invalid render_format: {config.render_format}"
                )


class RangeValue(GeneratorType):
    generic_name = "range"

    def text_description(self, config: EngineConfig) -> str:
        match config.render_format:
            case "english":
                return "Range type"
            case "chinese":
                return "范围类型"
            case "python":
                return "range"
            case "typescript":
                return "Range"
            case _:
                raise NotImplementedError(
                    f"Invalid render_format: {config.render_format}"
                )


class SliceValue(AnyValue):
    generic_name = "slice"

    def text_description(self, config: EngineConfig) -> str:
        match config.render_format:
            case "english":
                return "Slice type"
            case "chinese":
                return "切片类型"
            case "python":
                return "slice"
            case "typescript":
                return "Slice"
            case _:
                raise NotImplementedError(
                    f"Invalid render_format: {config.render_format}"
                )


class EllipsisValue(AnyValue):
    generic_name = "ellipsis"

    def text_description(self, config: EngineConfig) -> str:
        match config.render_format:
            case "english":
                return "Ellipsis type"
            case "chinese":
                return "省略类型"
            case "python":
                return "ellipsis"
            case "typescript":
                return "Ellipsis"
            case _:
                raise NotImplementedError(
                    f"Invalid render_format: {config.render_format}"
                )


class TensorValue(AnyValue):
    generic_name = "tensor"
    shape: List[int]
    dtype: str

    def text_description(self, config: EngineConfig) -> str:
        match config.render_format:
            case "english":
                return f"Tensor ({self.dtype}) of shape {self.shape}"
            case "chinese":
                return f"张量 ({self.dtype}) 形状为 {self.shape}"
            case "python":
                return f"tensor({self.dtype}, shape={self.shape})"
            case "typescript":
                return f"Tensor<{self.dtype}, {self.shape}>"
            case _:
                raise NotImplementedError(
                    f"Invalid render_format: {config.render_format}"
                )


class BytesValue(AnyValue):
    type = bytes
    generic_name = "bytes"

    def text_description(self, config: EngineConfig) -> str:
        match config.render_format:
            case "english":
                return "Bytes type"
            case "chinese":
                return "字节类型"
            case "python":
                return "bytes"
            case "typescript":
                return "Bytes"
            case _:
                raise NotImplementedError(
                    f"Invalid render_format: {config.render_format}"
                )


class NoneValue(AnyValue):
    type = None
    generic_name = "none"

    def text_description(self, config: EngineConfig) -> str:
        match config.render_format:
            case "english":
                return "None type"
            case "chinese":
                return "无类型"
            case "python":
                return "none"
            case "typescript":
                return "None"
            case _:
                raise NotImplementedError(
                    f"Invalid render_format: {config.render_format}"
                )
