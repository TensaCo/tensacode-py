from typing import Any, ClassVar, Optional, Mapping, Sequence
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
    input_node_type: ClassVar[type[Node]]

    def handler_match_score(
        self,
        input: object,
        /,
        *args,
        **kwargs,
    ) -> float:
        input_node = parse_node(input)
        dist = inheritance_distance(sub=input_node, parent=self.node_type)
        dist = float("inf") if dist is None else dist
        score = -1 * dist
        parent_score = super().handler_match_score(input, *args, **kwargs)
        return score + parent_score


class ModifyAtomicValueOp(BaseModifyOp):
    input_node_type: ClassVar[type[AtomicValueNode]] = AtomicValueNode

    def _execute(
        self,
        input_atom: object,
        /,
        *args,
        engine: BaseEngine,
        **kwargs,
    ) -> Any:
        engine.info(old_value=input_atom, type=type(input_atom))
        new_value = engine.decode(
            type=type(input_atom),
            prompt="Modify atomic value",
            current_value=input_atom,
            **kwargs,
        )
        engine.info(new_value=new_value)
        return new_value


class ModifySequenceValueOp(BaseModifyOp):
    input_node_type: ClassVar[type[SequenceNode]] = SequenceNode
    MODIFY_SEQ_OPTIONS = ["append", "insert", "pop", "remove", "modify"]

    def _execute(
        self,
        input_seq: Sequence[object],
        /,
        engine: BaseEngine,
        total_rounds: int = 10,
        **kwargs,
    ) -> Any:
        sequence_type = type(input_seq)
        element_type = greatest_common_type(input_seq)

        step = 0
        while engine.decide("Continue modifying sequence?") and (
            total_rounds is None or step < total_rounds
        ):
            step += 1
            engine.info(seq=input_seq)

            with engine.scope(
                action=f"modify ({step}/{total_rounds})",
                step=step,
                total_steps=total_rounds,
                input_seq=input_seq,
                type=sequence_type,
            ):
                action = engine.select(self.MODIFY_SEQ_OPTIONS, prompt="Select action")
                engine.info(action=action)
                match action:
                    case "append":
                        new_element = engine.decode(
                            type=element_type,
                            prompt="Create new element to append",
                            **kwargs,
                        )
                        input_seq.append(new_element)
                        engine.info(new_element=new_element)

                    case "insert":
                        index = engine.select(
                            list(range(len(input_seq))),
                            prompt="Select index to insert",
                        )
                        new_element = engine.decode(
                            type=element_type,
                            prompt="Insert new element",
                            index=index,
                            **kwargs,
                        )
                        input_seq.insert(index, new_element)
                        engine.info(new_element=new_element, index=index)

                    case "pop":
                        if not input_seq:
                            engine.feedback("Sequence is empty, cannot pop elements")
                            continue
                        popped_element = input_seq.pop()
                        engine.info(popped_element=popped_element)

                    case "remove":
                        if not input_seq:
                            engine.feedback("Sequence is empty, cannot remove elements")
                            continue
                        index = engine.select(
                            list(range(len(input_seq))),
                            prompt="Select index to remove",
                        )
                        removed_element = input_seq.pop(index)
                        engine.info(removed_element=removed_element, index=index)

                    case "modify":
                        if not input_seq:
                            engine.feedback("Sequence is empty, cannot modify elements")
                            continue
                        index = engine.select(
                            list(range(len(input_seq))),
                            prompt="Select index to modify",
                        )
                        current_value = input_seq[index]
                        new_value = engine.decode(
                            type=element_type,
                            prompt=f"Modify element at index {index}",
                            index=index,
                            current_value=current_value,
                            **kwargs,
                        )
                        input_seq[index] = new_value
                        engine.info(
                            index=index,
                            current_value=current_value,
                            new_value=new_value,
                        )
                    case _:
                        engine.feedback(
                            f"Invalid action: {action}. Must be one of {self.MODIFY_SEQ_OPTIONS}"
                        )

        return sequence_type(input_seq)


class ModifyMappingValueOp(BaseModifyOp):
    input_node_type: ClassVar[type[MappingNode]] = MappingNode

    MODIFY_MAP_OPTIONS = ["add", "remove", "modify"]

    def _execute(
        self,
        input_map: Mapping[str, object],
        /,
        *args,
        engine: BaseEngine,
        **kwargs,
    ) -> Any:
        mapping_type = type(input_map)
        input_dict = dict(input_map)
        element_type = greatest_common_type(input_dict.values())
        step = 0
        total_rounds = kwargs.get("total_rounds", 10)
        while engine.decide("Continue modifying mapping?") and (
            total_rounds is None or step < total_rounds
        ):
            step += 1
            engine.info(current_dict=input_dict)

            with engine.scope(
                action=f"modify ({step}/{total_rounds})",
                step=step,
                total_steps=total_rounds,
                input_map=input_dict,
                type=mapping_type,
            ):
                action = engine.select(
                    self.MODIFY_MAP_OPTIONS,
                    prompt="Select action to perform on the mapping",
                )
                engine.info(action=action)

                match action:

                    case "add":
                        key = engine.select(
                            prompt="Enter new key",
                            options=list(input_dict.keys()),
                        )
                        value = engine.decode(
                            type=element_type,
                            prompt=f"Create new value for key '{key}'",
                            **kwargs,
                        )
                        input_dict[key] = value

                        engine.info(new_key=key, new_value=value)
                    case "remove":
                        key = engine.select(
                            prompt="Select key to remove",
                            options=list(input_dict.keys()),
                        )
                        del input_dict[key]
                        engine.info(removed_key=key)

                    case "modify":
                        key = engine.select(
                            prompt="Select key to modify",
                            options=list(input_dict.keys()),
                        )
                        current_value = input_dict[key]
                        value = engine.decode(
                            type=element_type,
                            prompt=f"Modify value for key '{key}'",
                            current_value=current_value,
                            **kwargs,
                        )
                        input_dict[key] = value
                        engine.info(
                            modified_key=key,
                            current_value=current_value,
                            new_value=value,
                        )

                    case _:
                        engine.warn(f"Unknown action: {action}")

        return mapping_type(input_dict)


class ModifyCompositeValueOp(BaseModifyOp):
    input_node_type: ClassVar[type[CompositeValueNode]] = CompositeValueNode


class ModifyCompositeValueOp(BaseModifyOp):
    input_node_type: ClassVar[type[CompositeValueNode]] = CompositeValueNode

    MODIFY_COMPOSITE_OPTIONS = ["add", "remove", "modify"]

    def _execute(
        self,
        input_composite: object,
        /,
        engine: BaseEngine,
        **kwargs,
    ) -> Any:
        composite_type = type(input_composite)
        attributes = {
            attr: getattr(input_composite, attr)
            for attr in dir(input_composite)
            if not attr.startswith("_")
        }
        step = 0
        total_rounds = kwargs.get("total_rounds", 10)

        while engine.decide("Continue modifying composite object?") and (
            total_rounds is None or step < total_rounds
        ):
            step += 1
            engine.info(current_attributes=attributes)

            with engine.scope(
                action=f"modify ({step}/{total_rounds})",
                step=step,
                total_steps=total_rounds,
                input_composite=attributes,
                type=composite_type,
            ):
                action = engine.select(
                    self.MODIFY_COMPOSITE_OPTIONS,
                    prompt="Select action to perform on the composite object",
                )
                engine.info(action=action)

                match action:
                    case "add":
                        attr_name = engine.select(
                            prompt="Enter new attribute name",
                            options=list(attributes.keys()),
                        )
                        attr_value = engine.decode(
                            type=object,
                            prompt=f"Create new value for attribute '{attr_name}'",
                            **kwargs,
                        )
                        attributes[attr_name] = attr_value
                        setattr(input_composite, attr_name, attr_value)
                        engine.info(new_attribute=attr_name, new_value=attr_value)

                    case "remove":
                        attr_name = engine.select(
                            prompt="Select attribute to remove",
                            options=list(attributes.keys()),
                        )
                        del attributes[attr_name]
                        delattr(input_composite, attr_name)
                        engine.info(removed_attribute=attr_name)

                    case "modify":
                        attr_name = engine.select(
                            prompt="Select attribute to modify",
                            options=list(attributes.keys()),
                        )
                        current_value = attributes[attr_name]
                        new_value = engine.decode(
                            type=type(current_value),
                            prompt=f"Modify value for attribute '{attr_name}'",
                            current_value=current_value,
                            **kwargs,
                        )
                        attributes[attr_name] = new_value
                        setattr(input_composite, attr_name, new_value)
                        engine.info(
                            modified_attribute=attr_name,
                            current_value=current_value,
                            new_value=new_value,
                        )

                    case _:
                        engine.warn(f"Unknown action: {action}")

        return input_composite
