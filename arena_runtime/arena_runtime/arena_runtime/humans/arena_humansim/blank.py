"""Fresh default ROS messages at a fraction of the generated constructor's cost."""

import array
import typing
from collections.abc import Callable

import numpy as np


def blank_factory[M](cls: type[M]) -> Callable[[], M]:
    """A function returning a new message equal to `cls()`, value for value and type for type.

    The generated constructor reads every field from its keywords and stores it through the field's property; this one
    copies a default instance's slots straight into a new one: immutable values are shared, sequences and nested messages
    are fresh, so a message it returns shares nothing mutable with another one.
    """
    template = cls()
    new = cls.__new__
    shared: list[tuple[str, object]] = []
    nested: list[tuple[str, Callable[[], object]]] = []
    lists: list[str] = []
    arrays: list[tuple[str, array.array | np.ndarray]] = []
    for slot in typing.cast(tuple[str, ...], cls.__slots__):  # pyright: ignore[reportAttributeAccessIssue]
        value = getattr(template, slot)
        if hasattr(type(value), "get_fields_and_field_types"):
            nested.append((slot, blank_factory(type(value))))
        elif isinstance(value, list):
            if value:
                raise TypeError(f"{cls.__name__}.{slot} defaults to a non-empty list")
            lists.append(slot)
        elif isinstance(value, (array.array, np.ndarray)):
            arrays.append((slot, value))
        elif value is None or isinstance(value, (bool, int, float, str, bytes)):
            shared.append((slot, value))
        else:
            raise TypeError(f"{cls.__name__}.{slot} defaults to a {type(value).__name__}, which blank_factory does not copy")

    def blank() -> M:
        msg = new(cls)
        for slot, value in shared:
            setattr(msg, slot, value)
        for slot, make in nested:
            setattr(msg, slot, make())
        for slot in lists:
            setattr(msg, slot, [])
        for slot, value in arrays:
            setattr(msg, slot, value.copy() if isinstance(value, np.ndarray) else array.array(value.typecode, value))
        return msg

    return blank
