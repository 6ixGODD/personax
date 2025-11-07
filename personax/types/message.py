from __future__ import annotations

import typing as t

import pydantic as pydt

from personax.types import BaseModel
from personax.types import BaseSchema


class Message(BaseSchema):
    """Individual message in a conversation.

    Represents a single conversational turn with role, text content, and
    optional binary image data. Messages form the core of the conversation
    history processed by ContextSystems.

    Attributes:
        role: The speaker role ("system", "user", or "assistant").
        content: The text content of the message. May be None for image-only messages.
        image: Optional binary image data (e.g., JPEG/PNG bytes). Typically
            processed by vision-based ContextSystems and replaced with textual
            descriptions during the postprocess phase.

    Example:
        ```python
        # Text-only message
        user_msg = Message(
            role="user",
            content="What should I wear today?"
        )

        # Message with image (e.g., tongue diagnosis)
        image_msg = Message(
            role="user",
            content="Please analyze this",
            image=image_bytes  # Raw image data
        )

        # System message (for providing instructions)
        system_msg = Message(
            role="system",
            content="You are a helpful TCM health assistant."
        )
        ```
    """

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
    """Collection of conversation messages with validation.

    Encapsulates a sequence of messages forming a conversation, with built-in
    validation to ensure proper conversational structure (alternating roles,
    valid first/last messages, etc.).

    Attributes:
        messages: Sequence of Message objects representing the conversation.

    Example:
        ```python
        # Valid conversation
        messages = Messages(messages=[
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi! How can I help?"),
            Message(role="user", content="What's the weather?"),
        ])

        # Invalid: consecutive user messages (raises ValidationError)
        invalid = Messages(messages=[
            Message(role="user", content="Hello"),
            Message(role="user", content="Are you there?"),  # Error!
        ])

        # Invalid: ends with assistant (raises ValidationError)
        invalid = Messages(messages=[
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi!"),  # Error: must end with user
        ])
        ```
    """

    messages: t.Sequence[Message]

    def _validate(
        self,
        *,
        allow_system: bool = False,
        first_role: t.Literal["system", "user", "assistant"] = "user",
        last_role: t.Literal["system", "user", "assistant"] = "user",
    ) -> None:
        """Validate message sequence structure.

        Ensures:
        - Messages are not empty
        - First message has the correct role
        - Last message has the correct role
        - Messages alternate between user and assistant (except system if allowed)
        - All roles are valid

        Args:
            allow_system: Whether to allow system messages in the sequence.
            first_role: Required role for the first message.
            last_role: Required role for the last message.

        Raises:
            ValueError: If validation fails.
        """
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
        """Pydantic validator ensuring message structure.

        Validates that:
        - First message is from user
        - Last message is from user
        - Messages alternate between user and assistant
        - No system messages (allow_system=False)

        Returns:
            Self after successful validation.

        Raises:
            ValueError: If validation fails.
        """
        self._validate(allow_system=False, first_role="user", last_role="user")
        return self
