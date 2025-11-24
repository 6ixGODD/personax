from __future__ import annotations

import typing as t

from personax.types.message import Message


class Context(t.NamedTuple):
    """Shared context object passed through the ContextSystem pipeline.

    Encapsulates the conversation state and accumulated contextual data as
    it flows through multiple ContextSystems in a ContextCompose. Each system
    can inspect, modify, and enrich this context during the three-phase
    lifecycle (preprocess → build → postprocess).

    The context dictionary serves as a shared namespace where each ContextSystem
    stores its built data under its unique __key__. This enables downstream
    systems to access and build upon information from preceding systems.

    Attributes:
        messages: List of conversation messages. Systems can inspect these to
            extract queries, detect images, or understand conversation flow.
            The postprocess phase allows systems to modify messages (e.g.,
            replacing images with textual descriptions).
        context: Shared dictionary for storing system-specific data and
            metadata. Each ContextSystem stores its built content here under
            its __key__. Also used for passing external data like user profiles
            via extras.
    """

    messages: list[Message]
    context: dict[str, t.Any]
