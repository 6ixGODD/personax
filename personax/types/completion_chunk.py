from __future__ import annotations

import typing as t

from personax.types.completion import CompletionMessage
from personax.types.usage import Usage

CompletionChunkDelta = CompletionMessage


class CompletionChunk(t.NamedTuple):
    """Incremental LLM response chunk for streaming.

    Represents a single piece of a streaming completion response. Multiple
    chunks are emitted sequentially to build up the complete response.

    Attributes:
        id: Unique identifier for the completion (shared across all chunks).
        delta: Incremental content in this chunk. May contain partial text,
            refusal, or reasoning updates.
        finish_reason: Completion end reason, only set in the final chunk.
            None for all intermediate chunks.
        created: Unix timestamp when the completion was created (shared
            across all chunks).
        model: Model identifier (metadata field, shared across all chunks).
        usage: Token usage statistics. Typically only provided in the final
            chunk, None for intermediate chunks.

    Note:
        - Only delta.content changes between chunks (incremental text)
        - finish_reason is None until the final chunk
        - usage is typically None until the final chunk
        - id, created, and model remain constant across all chunks
    """

    id: str
    delta: CompletionChunkDelta
    finish_reason: t.Literal["stop", "length", "content_filter"] | None
    created: int
    model: str
    usage: Usage | None
