from __future__ import annotations

import typing as t

import pydantic as pydt

from personax.types import BaseModel
from personax.types import BaseSchema

class Message(BaseSchema):
    __slots__ = ("role", "content", "image")

    def __init__(
        self,
        *,
        role: t.Literal["user", "assistant"],
        content: str | None,
        image: bytes | None = None,
    ) -> None:
        super().__init__()
        self.role = role
        self.content = content
        self.image = image


class Messages(BaseModel):
    messages: t.Iterable[Message]

    @pydt.model_validator(mode="after")
    def val_messages(self) -> t.Self:
        iterator = iter(self.messages)

        try:
            first = next(iterator)
        except StopIteration:
            raise ValueError("Messages cannot be empty")

        if first.role not in ("user", "assistant"):
            raise ValueError(f"Role must be 'user' or 'assistant', got {first.role!r}")

        if first.role != "user":
            raise ValueError("First message must be from 'user'")

        prev = first
        last = first
        for i, curr in enumerate(iterator, start=1):
            if curr.role not in ("user", "assistant"):
                raise ValueError(f"Role must be 'user' or 'assistant', got {curr.role!r}")
            if prev.role == curr.role:
                raise ValueError("Messages must alternate between 'user' and 'assistant'")
            prev = curr
            last = curr

        if last.role != "user":
            raise ValueError("Last message must be from 'user'")

        return self
