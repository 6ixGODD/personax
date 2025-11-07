from __future__ import annotations

import typing as t

from personax.types import BaseSchema
from personax.types.completion import CompletionMessage
from personax.types.usage import Usage

CompletionChunkDelta = CompletionMessage


class CompletionChunk(BaseSchema):
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

    Example:
        ```python
        stream = await system.complete(
            messages, model="gpt-4", stream=True
        )

        full_content = ""
        async for chunk in stream:
            if chunk.delta.content:
                print(chunk.delta.content, end="", flush=True)
                full_content += chunk.delta.content

            if chunk.finish_reason:
                print(f"\nCompleted: {chunk.finish_reason}")

            if chunk.usage:
                print(f"Total tokens: {chunk.usage.total_tokens}")


        # Stream processing with operations
        stream = await system.complete(
            messages, model="gpt-4", stream=True
        )

        # Collect only content chunks
        content_stream = stream.filter(
            lambda c: c.delta.content is not None
        )

        # Take first 10 chunks
        limited = content_stream.take(10)

        async for chunk in limited:
            print(chunk.delta.content)
        ```

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

    __slots__ = ("created", "delta", "finish_reason", "id", "model", "usage")

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
