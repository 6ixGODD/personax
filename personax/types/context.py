from __future__ import annotations

import typing as t

from personax.types import BaseSchema
from personax.types.message import Message


class Context(BaseSchema):
    messages: list[Message]
    context: dict[str, t.Any]

    __slots__ = ("context", "messages")

    def __init__(self, *, messages: list[Message], context: dict[str, t.Any] | None = None):
        self.messages = messages
        self.context = context or {}
