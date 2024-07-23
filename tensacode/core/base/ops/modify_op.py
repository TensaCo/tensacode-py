from typing import Any, ClassVar, Optional, Mapping
import inspect
from typing_extensions import Self

from tensacode.core.base.base_engine import BaseEngine
from tensacode.internal.protocols.latent import LatentType
from tensacode.core.base.ops.base_op import BaseOp
from tensacode.core.base.ops.select_op import SelectOp
from tensacode.core.base.ops.decode_op import DecodeOp
from tensacode.core.base.ops.decide_op import DecideOp
from tensacode.internal.protocols.encoded import Encoded
from tensacode.internal.tcir.nodes import (
    SequenceNode,
    MappingNode,
    AtomicValueNode,
    Node,
    CompositeValueNode,
)
from tensacode.internal.tcir.parse import parse_node
from tensacode.internal.utils.misc import inheritance_distance, greatest_common_type


@BaseEngine.register_op_class_for_all_class_instances
class BaseModifyOp(BaseOp):
    """
    ModifyOp: An operation class for modifying objects in a BaseEngine environment.

    This class extends BaseOp to provide functionality for iteratively modifying
    fields of an input object. It uses various engine operations like decide,
    select, and decode to interactively modify the object's attributes or dictionary items.

    Attributes:
        op_name (ClassVar[str]): The name of the operation, set to "modify".
        object_type (ClassVar[type[object]]): The type of object this operation can modify, set to Any.
        latent_type (ClassVar[LatentType]): The latent type used by this operation, set to LatentType.
        engine_type (ClassVar[type[BaseEngine]]): The type of engine this operation is compatible with, set to BaseEngine.

    Methods:
        _execute(self, input: object, *, engine: BaseEngine, **kwargs) -> Any:
            The core method that performs the modification operation.

    Usage:
        This operation can be used in two ways:

        1. Through the BaseEngine.modify() method:

        >>> engine = BaseEngine()
        >>> modified_object = engine.modify(my_object)

        2. Directly, by instantiating the ModifyOp class:

        >>> engine = BaseEngine()
        >>> modify_op = ModifyOp()
        >>> modified_object = modify_op.execute(my_object, engine=engine)

        Or using the call syntax:

        >>> modified_object = modify_op(my_object, engine=engine)

    Examples:
        Example 1: Modifying a dictionary
        >>> engine = BaseEngine()
        >>> modify_op = ModifyOp()
        >>> my_dict = {"name": "Alice", "age": 30}
        >>> modified_dict = modify_op(my_dict, engine=engine)
        # The engine will interactively prompt for modifications

        Example 2: Modifying an object with attributes
        >>> class Person:
        ...     def __init__(self, name, age):
        ...         self.name = name
        ...         self.age = age
        >>> person = Person("Bob", 25)
        >>> modified_person = modify_op.execute(person, engine=engine)
        # The engine will interactively prompt for modifications

        Example 3: Using with custom engine and context
        >>> class MyEngine(BaseEngine):
        ...     pass
        >>> my_engine = MyEngine()
        >>> context = {"modification_limit": 3}
        >>> config = {"strict_mode": True}
        >>> result = modify_op(my_object, engine=my_engine, context=context, config=config)

    The modification process:
    1. Repeatedly asks the engine to decide whether to continue modifying.
    2. For each modification:
        a. Selects a field to modify using engine.select().
        b. Retrieves the current value and type of the selected field.
        c. Uses engine.decode() to generate a new value for the field.
        d. Updates the input object with the new value.

    Note:
        - This operation works with both dictionary-like objects and objects with attributes.
        - It uses introspection to determine field types when possible.
        - The operation provides feedback and logging through the engine's context.

    See Also:
        BaseOp: The base class for all operations.
        BaseEngine.modify: The engine method that typically invokes this operation.
    """

    op_name: ClassVar[str] = "modify"
    latent_type: ClassVar[LatentType] = LatentType
    engine_type: ClassVar[type[BaseEngine]] = BaseEngine


class ModifyAtomicValueOp(BaseModifyOp):
    def handler_match_score(
        self,
        input: object,
        *args,
        engine_type: BaseEngine,
        latent_type: LatentType,
        operator_type: type | None = None,
        operator_name: str | None = None,
        **kwargs,
    ) -> int:
        input_node = parse_node(input)
        dist = inheritance_distance(sub=input_node, parent=AtomicValueNode)
        if not dist:
            return None
        return -1 * dist + super().handler_match_score(
            *args,
            engine_type=engine_type,
            latent_type=latent_type,
            operator_type=operator_type,
            operator_name=operator_name,
            **kwargs,
        )

    @BaseEngine.trace
    def _execute(
        self,
        input: object,
        *,
        engine: BaseEngine,
        **kwargs,
    ) -> Any:
        engine.info(old_value=input, type=type(input))
        new_value = engine.decode(
            type=type(input),
            prompt="Modify atomic value",
            **kwargs,
        )
        engine.info(new_value=new_value)
        return new_value


class ModifySequenceValueOp(BaseModifyOp):
    def handler_match_score(
        self,
        input: object,
        *args,
        engine_type: BaseEngine,
        latent_type: LatentType,
        operator_type: type | None = None,
        operator_name: str | None = None,
        **kwargs,
    ) -> int:
        input_node = parse_node(input)
        dist = inheritance_distance(sub=input_node, parent=SequenceNode)
        if not dist:
            return None
        return -1 * dist + super().handler_match_score(
            *args,
            engine_type=engine_type,
            latent_type=latent_type,
            operator_type=operator_type,
            operator_name=operator_name,
            **kwargs,
        )

    @BaseEngine.trace
    def _execute(
        self,
        input: object,
        *,
        engine: BaseEngine,
        **kwargs,
    ) -> Any:
        sequence_type = type(input)
        element_type = greatest_common_type(input)

        while engine.decide("Continue modifying sequence?"):
            match engine.select(["add", "remove", "modify"], prompt="Select action"):
                case "add":
                    new_element = engine.decode(
                        type=element_type,
                        prompt="Add new element",
                        **kwargs,
                    )
                    input.append(new_element)
                    engine.info(action="add", new_element=new_element)

                case "remove":
                    if not input:
                        engine.feedback("Sequence is empty, cannot remove elements")
                        continue
                    index = engine.select(
                        range(len(input)), prompt="Select index to remove"
                    )
                    removed_element = input.pop(index)
                    engine.info(
                        action="remove", removed_element=removed_element, index=index
                    )

                case "modify":
                    if not input:
                        engine.feedback("Sequence is empty, cannot modify elements")
                        continue
                    index = engine.select(
                        range(len(input)), prompt="Select index to modify"
                    )
                    old_value = input[index]
                    new_value = engine.decode(
                        type=element_type,
                        prompt=f"Modify element at index {index}",
                        **kwargs,
                    )
                    input[index] = new_value
                    engine.info(
                        action="modify",
                        index=index,
                        old_value=old_value,
                        new_value=new_value,
                    )
                case _:
                    engine.feedback(f"Invalid action: {action}")

        return sequence_type(input)


class ModifyMappingValueOp(BaseModifyOp):
    def handler_match_score(
        self,
        input: object,
        *args,
        engine_type: BaseEngine,
        latent_type: LatentType,
        operator_type: type | None = None,
        operator_name: str | None = None,
        **kwargs,
    ) -> int:
        input_node = parse_node(input)
        dist = inheritance_distance(sub=input_node, parent=MappingNode)
        if not dist:
            return None
        return -1 * dist + super().handler_match_score(
            *args,
            engine_type=engine_type,
            latent_type=latent_type,
            operator_type=operator_type,
            operator_name=operator_name,
            **kwargs,
        )

    @BaseEngine.trace
    def _execute(
        self,
        input: object,
        *,
        engine: BaseEngine,
        **kwargs,
    ) -> Any:

        i: int = 0
        while engine.decide("Continue modifying object?"):
            engine.log.info(f"Pass {i}")
            field = engine.select(input, prompt="Select field to modify")
            old_value = input.get(field, None)
            if not old_value:
                engine.feedback(f"Field {field} not found in input")
                continue
            old_type = type(old_value)

            engine.info(field=field, old_value=old_value, type=old_type)

            value = engine.decode(
                type=field_annotation, prompt=f"Modify {field} value", **kwargs
            )
            engine.info(field=field, new_value=value)
            input[field] = value
            i += 1
        return input


class ModifyCompositeValueOp(BaseModifyOp):
    def handler_match_score(
        self,
        input: object,
        *args,
        engine_type: BaseEngine,
        latent_type: LatentType,
        operator_type: type | None = None,
        operator_name: str | None = None,
        **kwargs,
    ) -> int:
        input_node = parse_node(input)
        dist = inheritance_distance(sub=input_node, parent=CompositeValueNode)
        if not dist:
            return None
        return -1 * dist + super().handler_match_score(
            *args,
            engine_type=engine_type,
            latent_type=latent_type,
            operator_type=operator_type,
            operator_name=operator_name,
            **kwargs,
        )

    @BaseEngine.trace
    def _execute(
        self,
        input: object,
        *,
        engine: BaseEngine,
        **kwargs,
    ) -> Any:

        i: int = 0
        while engine.decide("Continue modifying object?"):
            engine.log.info(f"Pass {i}")
            field = engine.select(input, prompt="Select field to modify")
            old_value = getattr(input, field)
            if old_value is None:
                engine.feedback(f"Field {field} not found in input")
                continue
            old_type = type(old_value)
            engine.info(field=field, old_value=old_value, type=old_type)

            engine.info(
                field=field,
                old_value=old_value,
                type=old_type,
            )

            value = engine.decode(
                type=old_type,
                prompt=f"Modify {field} value",
                **kwargs,
            )
            engine.info(field=field, new_value=value)
            if isinstance(input, Mapping):
                input[field] = value
            else:
                setattr(input, field, value)

        return input
