from __future__ import annotations

import typing as t

from personax.types import BaseSchema
from personax.types.message import Message


class Context(BaseSchema):
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

    Example:
        ```python
        # Initial context with user profile data
        context = Context(
            messages=[
                Message(role="user", content="What's the weather?"),
            ],
            context={
                "profile.info": {"location": "San Francisco", "timezone": "PST"}
            }
        )

        # After ProfileContextSystem build phase
        # context.context = {
        #     "profile.info": {...},
        #     "profile": {
        #         "location": "San Francisco",
        #         "timestamp": "2025-11-07T05:54:44Z",
        #         ...
        #     }
        # }

        # After WeatherContextSystem build phase (can access profile data)
        # context.context = {
        #     "profile.info": {...},
        #     "profile": {...},
        #     "weather": {
        #         "temperature": 15.5,
        #         "condition": "Cloudy",
        #         "location": "San Francisco"  # Extracted from profile
        #     }
        # }
        ```
    """

    messages: list[Message]
    context: dict[str, t.Any]

    __slots__ = ("context", "messages")

    def __init__(self, *, messages: list[Message], context: dict[str, t.Any] | None = None):
        self.messages = messages
        self.context = context or {}
