from __future__ import annotations

import typing as t

import pydantic as pydt

from personax.types import BaseSchema
from personax.types.message import Messages as RawMessages


class Message(BaseSchema):
    role: t.Literal["user", "assistant", "system"]
    content: str

    __slots__ = ("content", "role")

    def __init__(
        self,
        *,
        role: t.Literal["user", "assistant", "system"],
        content: str,
    ) -> None:
        super().__init__()
        self.role = role
        self.content = content


class Messages(RawMessages):
    messages: t.Iterable[Message]  # type: ignore[assignment]

    @pydt.model_validator(mode="after")
    def validate_messages(self) -> t.Self:
        self._validate(allow_system=True, first_role="system", last_role="user")
        return self

    @classmethod
    def from_raws(cls, raws: RawMessages, sys_prompt: str) -> t.Self:
        return cls.model_construct(
            messages=[
                Message(role="system", content=sys_prompt),
                *[Message(role=msg.role, content=msg.content or "") for msg in raws.messages],
            ]
        )
