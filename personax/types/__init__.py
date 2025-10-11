from __future__ import annotations

import os
import typing as t

import pydantic as pydt


class BaseSchema:
    __slots__ = ()  # type: t.Collection[str]

    def dumps(self) -> dict[str, object]:
        return {slot: self._dumps(getattr(self, slot, None)) for slot in self.__slots__}

    def _dumps(self, v: object, /) -> object:
        if isinstance(v, BaseSchema):
            return v.dumps()
        elif isinstance(v, (list, tuple)):
            return [self._dumps(item) for item in v]
        else:
            return v

    def __eq__(self, other: object, /) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return all(
            getattr(self, slot, None) == getattr(other, slot, None) for slot in self.__slots__)

    def __repr__(self) -> str:
        attrs = ", ".join(f"{slot}={getattr(self, slot, None)!r}" for slot in self.__slots__)
        return f"{self.__class__.__name__}({attrs})"

    __str__ = __repr__

    def __getstate__(self) -> tuple[object, ...]:  # for pickle optimization
        return tuple(getattr(self, slot, None) for slot in self.__slots__)

    def __setstate__(self, state: tuple[object, ...]):
        for slot, value in zip(self.__slots__, state):
            setattr(self, slot, value)


class BaseModel(pydt.BaseModel):
    model_config: t.ClassVar[pydt.ConfigDict] = pydt.ConfigDict(
        validate_assignment=True,  # Validate on assignment
        validate_default=False,  # Do not validate default values
        extra="forbid",  # Disallow extra fields
        arbitrary_types_allowed=True,  # Allow arbitrary types
        populate_by_name=True,  # Allow population by field name
        use_enum_values=True,  # Use enum values directly
        frozen=True,  # Make the model immutable
    )


PathLikes: t.TypeAlias = str | os.PathLike[str]
