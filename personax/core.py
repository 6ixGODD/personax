from __future__ import annotations

import typing as t

from personax.completion import CompletionSystem
from personax.context import ContextCompose
from personax.tools import BaseToolType
from personax.types.completion import Completion
from personax.types.completion_chunk import CompletionChunk
from personax.types.message import Messages
from personax.types.stream import AsyncStream
from personax.utils import UNSET
from personax.utils import AsyncContextMixin
from personax.utils import Unset
from personax.utils import classproperty


def build_id(
    name: str,
    version: t.Literal["latest"] | str = "latest",
    scenario: t.Literal["default"] | str = "default",
) -> str:
    """Build a unique identifier string for a persona.

    Args:
        name: The base name of the persona
        version: Version string, defaults to "latest"
        scenario: Scenario string, defaults to "default"

    Returns:
        A formatted ID string in the format: name[-version][@scenario]
    """
    return (
        f"{name}"
        + (f"-{version}" if version != "latest" else "")
        + (f"@{scenario}" if scenario != "default" else "")
    )


class Core(AsyncContextMixin):
    __slots__ = ("completion", "context", "model_id", "toolset")

    def __init__(
        self,
        *,
        context: ContextCompose,
        toolset: t.Iterable[BaseToolType] = (),
        completion: CompletionSystem,
        model_id: str,
    ) -> None:
        self.context = context
        self.toolset = toolset
        self.completion = completion
        self.model_id = model_id

    async def init(self) -> None:
        await self.completion.init()
        await self.context.init()

    async def close(self) -> None:
        await self.completion.close()
        await self.context.close()

    async def complete(
        self,
        messages: Messages,
        *,
        chatcmpl_id: str | Unset = UNSET,
        stream: bool = False,
        max_completion_tokens: int | Unset = UNSET,
        prompt_cache_key: str | Unset = UNSET,
        extras: dict[str, t.Any] | None = None,
    ) -> Completion | AsyncStream[CompletionChunk]:
        """Complete a set of messages.

        Args:
            messages: The input messages to complete.
            chatcmpl_id: Optional ID for the chat completion.
            stream: Whether to stream the completion.
            max_completion_tokens: Maximum tokens for the completion.
            prompt_cache_key: Optional cache key for the prompt.
            extras: Additional context extras.

        Returns:
            A Completion object or an AsyncStream of CompletionChunk objects.
        """
        messages = await self.context.build(messages, extras)
        toolset = list(self.toolset)
        return await self.completion.complete(
            messages=messages,
            chatcmpl_id=chatcmpl_id,
            stream=stream,
            max_completion_tokens=max_completion_tokens,
            prompt_cache_key=prompt_cache_key,
            tools=toolset,
            model=self.model_id,
        )


class PersonaX(AsyncContextMixin):
    core: Core

    name: t.ClassVar[str]
    version: t.ClassVar[t.Literal["latest"] | str] = "latest"
    scenario: t.ClassVar[t.Literal["default"] | str] = "default"

    @classproperty
    def id(self) -> str:
        return build_id(self.name, self.version, self.scenario)

    def __init_subclass__(cls, **kwargs: t.Any) -> None:
        super().__init_subclass__(**kwargs)

        if not hasattr(cls, "name") or not isinstance(cls.name, str):
            raise NotImplementedError("Subclasses must define a 'name' class variable of type str.")

    def __str__(self) -> str:
        return f"<{self.__class__.__name__} {self.id}>"

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(name="{self.name}", version="{self.version}",'
            f'scenario="{self.scenario}")'
        )

    def __hash__(self) -> int:
        return hash(self.id)

    def __init__(self, core: Core) -> None:
        self.core = core

    async def init(self) -> None:
        await self.core.init()

    async def close(self) -> None:
        await self.core.close()

    async def complete(
        self,
        messages: Messages,
        *,
        chatcmpl_id: str | Unset = UNSET,
        stream: bool = False,
        max_completion_tokens: int | Unset = UNSET,
        prompt_cache_key: str | Unset = UNSET,
        extras: dict[str, t.Any] | None = None,
    ) -> Completion | AsyncStream[CompletionChunk]:
        return await self.core.complete(
            messages=messages,
            chatcmpl_id=chatcmpl_id,
            stream=stream,
            max_completion_tokens=max_completion_tokens,
            prompt_cache_key=prompt_cache_key,
            extras=extras,
        )
