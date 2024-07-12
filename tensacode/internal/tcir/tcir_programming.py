from __future__ import annotations

from abc import ABC, abstractmethod
import operator

from pydantic import BaseModel

from tensacode.internal.tcir.tcir_main import Any, Class, IsIterable, Type, Value
from tensacode.internal.tcir.tcir_values import Function, Module


class Statement(Any):
    name = "statement"
    statement: str


class DeclarativeStatement(Statement, ABC):
    name = "declarative_statement"
    declarative: str


class Comment(DeclarativeStatement):
    name = "comment"
    comment: str


class Annotation(DeclarativeStatement):
    name = "annotation"
    annotation: str


class ModuleDeclaration(DeclarativeStatement):
    name = "module_declaration"
    module_value: Module


class ClassDeclaration(DeclarativeStatement):
    name = "class_declaration"
    class_value: Class


class FunctionDeclaration(DeclarativeStatement):
    name = "function_declaration"
    function_value: Function


class Variable(Any):
    name = "variable"
    var_name: str
    type: Type
    value: Value


class VariableDeclaration(DeclarativeStatement):
    name = "variable_declaration"
    variable: Variable


class ImperativeStatement(Statement, ABC):
    name = "imperative_statement"


class AssignmentOperator(OperatorInvocation, BinaryOperator):
    name = "assignment_operator"
    var_name: Variable
    assignment: Expression


class FunctionCall(OperatorInvocation):
    name = "function_call"
    function: Function
    arguments: list[Value]


class Expression(BaseModel):
    name = "expression"
    expression: str


ExpressionBlock = list[Expression]


class ControlFlowStatement(ImperativeStatement, ABC):
    pass


class ConditionalBranching(ControlFlowStatement):
    class Branch(BaseModel):
        condition: Expression
        body: ExpressionBlock

    name = "conditional_branching"
    branches: list[Branch]


class WhileLoop(ControlFlowStatement):
    name = "while_loop"
    condition: Expression
    body: ExpressionBlock


class ForEachLoop(ControlFlowStatement):
    name = "for_each_loop"
    iterable: IsIterable
    body: ExpressionBlock


class OperatorInvocation(ImperativeStatement, ABC):
    name = "operator_invocation"
    operator: Operator

    @abstractmethod
    def arguments(self) -> list[Value]: ...


class UnaryOperatorInvocation(OperatorInvocation, ABC):
    name = "unary_operator_invocation"
    operator: UnaryOperator
    first_operand: Expression

    def arguments(self) -> list[Value]:
        return [self.first_operand]


class BinaryOperatorInvocation(OperatorInvocation, ABC):
    name = "binary_operator_invocation"
    operator: BinaryOperator
    first_operand: Expression
    second_operand: Expression

    def arguments(self) -> list[Value]:
        return [self.first_operand, self.second_operand]


class TernaryOperatorInvocation(OperatorInvocation, ABC):
    name = "ternary_operator_invocation"
    operator: TernaryOperator
    first_operand: Expression
    second_operand: Expression
    third_operand: Expression

    def arguments(self) -> list[Value]:
        return [self.first_operand, self.second_operand, self.third_operand]


class Operator(BaseModel, ABC):
    name: str
    arity: int

    @abstractmethod
    def apply(self, *args: Value) -> Value: ...

    @classmethod
    def register_operator(cls, name, arity, apply) -> type[Operator]:
        subcls = type(cls, (cls,), {"name": name, "arity": arity, "apply": apply})
        return subcls


class UnaryOperator(ABC):

    @abstractmethod
    def apply(self, first: Expression) -> Value: ...

    @classmethod
    def register_operator(cls, name, apply) -> type[UnaryOperator]:
        return Operator.register_operator(name, 1, apply)


class BinaryOperator(ABC):

    @abstractmethod
    def apply(self, first: Expression, second: Expression) -> Value: ...

    @classmethod
    def register_operator(cls, name, apply) -> type[BinaryOperator]:
        return Operator.register_operator(name, 2, apply)


class TernaryOperator(ABC):

    @abstractmethod
    def apply(
        self,
        first: Expression,
        second: Expression,
        third: Expression,
    ) -> Value: ...

    @classmethod
    def register_operator(cls, name, apply) -> type[TernaryOperator]:
        return Operator.register_operator(name, 3, apply)


class LogicalOperator(OperatorInvocation, ABC):
    name = "boolean_operator"
    operator: str


class AndOperator(LogicalOperator, BinaryOperator):
    name = "and"


class OrOperator(LogicalOperator, BinaryOperator):
    name = "or"


class NotOperator(LogicalOperator, UnaryOperator):
    name = "not"


class XorOperator(LogicalOperator, BinaryOperator):
    name = "xor"


class NandOperator(LogicalOperator, BinaryOperator):
    name = "nand"


class NorOperator(LogicalOperator, BinaryOperator):
    name = "nor"


class ImpliesOperator(LogicalOperator, BinaryOperator):
    name = "implies"


class BitwiseOperator(OperatorInvocation, ABC):
    name = "bitwise_operator"


class ShiftLeftOperator(BitwiseOperator, BinaryOperator):
    name = "shift_left"


class ShiftRightOperator(BitwiseOperator, BinaryOperator):
    name = "shift_right"


class BitwiseAndOperator(BitwiseOperator, BinaryOperator):
    name = "bitwise_and"


class BitwiseOrOperator(BitwiseOperator, BinaryOperator):
    name = "bitwise_or"


class BitwiseXorOperator(BitwiseOperator, BinaryOperator):
    name = "bitwise_xor"


class BitwiseNotOperator(BitwiseOperator, UnaryOperator):
    name = "bitwise_not"


class ArithmeticOperator(BitwiseOperator):
    name = "arithmetic_operator"


class AdditionOperator(ArithmeticOperator, BinaryOperator):
    name = "addition"


class SubtractionOperator(ArithmeticOperator, BinaryOperator):
    name = "subtraction"


class MultiplicationOperator(ArithmeticOperator, BinaryOperator):
    name = "multiplication"


class DivisionOperator(ArithmeticOperator, BinaryOperator):
    name = "division"


class ModulusOperator(ArithmeticOperator, BinaryOperator):
    name = "modulus"


class PowerOperator(ArithmeticOperator, BinaryOperator):
    name = "power"


class FloorDivisionOperator(ArithmeticOperator, BinaryOperator):
    name = "floor_division"


class ExponentiationOperator(ArithmeticOperator, BinaryOperator):
    name = "exponentiation"


class LogarithmOperator(ArithmeticOperator, BinaryOperator):
    name = "logarithm"


class SineOperator(ArithmeticOperator, UnaryOperator):
    name = "sine"


class CosineOperator(ArithmeticOperator, UnaryOperator):
    name = "cosine"


class TanOperator(ArithmeticOperator, UnaryOperator):
    name = "tan"


class AbsoluteOperator(ArithmeticOperator, UnaryOperator):
    name = "absolute"


class NegativeOperator(ArithmeticOperator, UnaryOperator):
    name = "negative"


class SquareRootOperator(ArithmeticOperator, UnaryOperator):
    name = "square_root"


class CubeRootOperator(ArithmeticOperator, UnaryOperator):
    name = "cube_root"


class ExponentialOperator(ArithmeticOperator, UnaryOperator):
    name = "exponential"


class LogarithmBase10Operator(ArithmeticOperator, UnaryOperator):
    name = "logarithm_base_10"


class LogarithmBase2Operator(ArithmeticOperator, UnaryOperator):
    name = "logarithm_base_2"


class SineHOperator(ArithmeticOperator, UnaryOperator):
    name = "sine_h"


class CosineHOperator(ArithmeticOperator, UnaryOperator):
    name = "cosine_h"


class TanHOperator(ArithmeticOperator, UnaryOperator):
    name = "tan_h"


class ArcSineOperator(ArithmeticOperator, UnaryOperator):
    name = "arc_sine"


class ArcCosineOperator(ArithmeticOperator, UnaryOperator):
    name = "arc_cosine"


class ArcTanOperator(ArithmeticOperator, UnaryOperator):
    name = "arc_tan"


class HyperbolicArcSineOperator(ArithmeticOperator, UnaryOperator):
    name = "hyperbolic_arc_sine"


class HyperbolicArcCosineOperator(ArithmeticOperator, UnaryOperator):
    name = "hyperbolic_arc_cosine"


class HyperbolicArcTanOperator(ArithmeticOperator, UnaryOperator):
    name = "hyperbolic_arc_tan"


class FactorialOperator(ArithmeticOperator, UnaryOperator):
    name = "factorial"


class CeilOperator(ArithmeticOperator, UnaryOperator):
    name = "ceil"


class FloorOperator(ArithmeticOperator, UnaryOperator):
    name = "floor"


class RoundOperator(ArithmeticOperator, UnaryOperator):
    name = "round"


class ContainerOperator(OperatorInvocation):
    name = "container_operator"


class MutableSequenceOperator(ContainerOperator):
    name = "mutable_sequence_operator"


class ImmutableSequenceOperator(ContainerOperator):
    name = "immutable_sequence_operator"


class IncludesOperator(MutableSequenceOperator, ImmutableSequenceOperator):
    name = "includes"


class LengthOperator(MutableSequenceOperator, ImmutableSequenceOperator):
    name = "length"


class DuplicateOperator(ImmutableSequenceOperator):
    name = "duplicate"


class IndexReadOperator(MutableSequenceOperator, ImmutableSequenceOperator):
    name = "index_read"
    index: list[Index]
    indexable: AnyValue


class IndexAssignOperator(MutableSequenceOperator):
    name = "index_assign"
    index: list[Index]
    indexable: AnyValue
    value: AnyValue


class IndexDeleteOperator(MutableSequenceOperator):
    name = "index_delete"
    index: list[Index]
    indexable: AnyValue
