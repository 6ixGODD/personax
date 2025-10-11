from __future__ import annotations

import typing as t

from personax.types import BaseSchema
from personax.types.completion import CompletionMessage
from personax.types.usage import Usage

CompletionChunkDelta = CompletionMessage


class CompletionChunk(BaseSchema):
    id: str
    delta: CompletionChunkDelta
    finish_reason: t.Literal["stop", "length", "content_filter"] | None
    created: int
    model: str
    usage: Usage | None

    __slots__ = ("id", "delta", "finish_reason", "created", "model", "usage")

    def __init__(
        self,
        *,
        id: str,  # pylint: disable=redefined-builtin
        delta: CompletionChunkDelta,
        finish_reason: t.Literal["stop", "length", "content_filter"] | None = None,
        created: int,
        model: str,
        usage: Usage | None = None,
    ) -> None:
        self.id = id
        self.delta = delta
        self.finish_reason = finish_reason
        self.created = created
        self.model = model
        self.usage = usage
