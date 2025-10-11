from __future__ import annotations

import typing as t

import pydantic as pydt

from personax.types import BaseModel
from personax.types import BaseSchema
from personax.types.message import Messages as RawMessages


class Message(BaseSchema):
    role: t.Literal["user", "assistant", "system"]
    content: str

    __slots__ = ("role", "content")

    def __init__(
        self,
        *,
        role: t.Literal["user", "assistant", "system"],
        content: str,
    ) -> None:
        super().__init__()
        self.role = role
        self.content = content


class Messages(BaseModel):
    messages: t.Iterable[Message]

    @pydt.model_validator(mode="after")
    def validate_messages(self) -> t.Self:
        iterator = iter(self.messages)

        try:
            first = next(iterator)
        except StopIteration:
            raise ValueError("Messages cannot be empty")

        if first.role != "system":
            raise ValueError("First message must be from 'system'")

        prev = first
        last = first
        for i, curr in enumerate(iterator, start=1):
            if curr.role == "system":
                raise ValueError("Only the first message can be 'system'")

            if curr.role not in ("user", "assistant"):
                raise ValueError(f"Role must be 'user' or 'assistant', got {curr.role!r}")

            if prev.role == curr.role:
                raise ValueError("Messages must alternate between 'user' and 'assistant'")

            prev = curr
            last = curr

        if last.role != "user":
            raise ValueError("Last message must be from 'user'")

        return self

    @classmethod
    def from_raws(cls, raws: RawMessages, sys_prompt: str) -> t.Self:
        return cls.model_construct(messages=[
            Message(role="system", content=sys_prompt),
            *[Message(role=msg.role, content=msg.content) for msg in raws.messages],
        ])
