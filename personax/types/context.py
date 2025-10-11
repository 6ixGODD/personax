from __future__ import annotations

import typing as t

from personax.types import BaseSchema
from personax.types.message import Message


class Context(BaseSchema):
    messages: t.List[Message]
    context: t.Dict[str, t.Any] | None

    __slots__ = ("messages", "context")

    def __init__(self, *, messages: t.List[Message], context: t.Dict[str, t.Any] | None = None):
        self.messages = messages
        self.context = context
