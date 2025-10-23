from __future__ import annotations

import typing as t

import pydantic as pydt

from personax.types import BaseModel
from personax.types import BaseSchema


class Message(BaseSchema):
    __slots__ = ("content", "image", "role")

    def __init__(
        self,
        *,
        role: t.Literal["system", "user", "assistant"],
        content: str | None,
        image: bytes | None = None,
    ) -> None:
        super().__init__()
        self.role = role
        self.content = content
        self.image = image


class Messages(BaseModel):
    messages: t.Sequence[Message]

    def _validate(
        self,
        *,
        allow_system: bool = False,
        first_role: t.Literal["system", "user", "assistant"] = "user",
        last_role: t.Literal["system", "user", "assistant"] = "user",
    ) -> None:
        messages = list(self.messages)
        iterator = iter(messages)
        try:
            first = next(iterator)
        except StopIteration as exc:
            raise ValueError("Messages cannot be empty") from exc

        if allow_system:
            if first.role not in ("user", "assistant", "system"):
                raise ValueError(f"Invalid role for first message: {first.role!r}")
        else:
            if first.role not in ("user", "assistant"):
                raise ValueError(f"Role must be 'user' or 'assistant', got {first.role!r}")
            if first.role != first_role:
                raise ValueError(f"First message must be {first_role}")

        prev = first
        last = first
        for curr in iterator:
            if curr.role not in ("user", "assistant") and not (
                allow_system and curr.role == "system"
            ):
                raise ValueError(f"Invalid role: {curr.role!r}")
            if prev.role == curr.role:
                raise ValueError("Messages must alternate between 'user' and 'assistant'")
            prev = curr
            last = curr

        if last.role != last_role:
            raise ValueError(f"Last message must be {last_role}")

    @pydt.model_validator(mode="after")
    def val_messages(self) -> t.Self:
        self._validate(allow_system=False, first_role="user", last_role="user")
        return self
