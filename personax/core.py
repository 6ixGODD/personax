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
    """Build a unique identifier for a PersonaX instance.

    Constructs a qualified name identifier combining the persona name,
    version, and scenario. Used for persona registration and lookup in
    orchestration systems.

    Format: `name[-version][@scenario]`
    - Version suffix is omitted for "latest"
    - Scenario suffix is omitted for "default"

    Args:
        name: Base name of the persona (e.g., "assistant", "persomna").
        version: Version identifier. Defaults to "latest".
        scenario: Scenario or use case identifier. Defaults to "default".

    Returns:
        Formatted identifier string.

    Example:
        ```python
        build_id("assistant")  # "assistant"
        build_id("assistant", "v2")  # "assistant-v2"
        build_id("assistant", "v2", "medical")  # "assistant-v2@medical"
        build_id("assistant", scenario="s2s")  # "assistant@s2s"
        ```
    """
    return (
        f"{name}"
        + (f"-{version}" if version != "latest" else "")
        + (f"@{scenario}" if scenario != "default" else "")
    )


class Core(AsyncContextMixin):
    """Core orchestrator for PersonaX completion pipeline.

    Integrates the three fundamental PersonaX components:
    1. ContextCompose: Builds enriched context from conversation messages
    2. CompletionSystem: Generates LLM responses
    3. Toolset: Provides callable functions for the LLM

    The core execution flow:
    1. Input messages → ContextCompose → enriched messages with system prompt
    2. Enriched messages + toolset → CompletionSystem → completion/stream

    This class manages the lifecycle of all components and provides a unified
    interface for generating completions.

    Attributes:
        context: Context composition system for building enriched prompts.
        toolset: Collection of tools available for LLM function calling.
        completion: Completion system for LLM text generation.
        model_id: Identifier for this model configuration.

    Args:
        context: Configured ContextCompose instance.
        toolset: Iterable of tool instances for function calling.
        completion: Configured CompletionSystem instance.
        model_id: Unique identifier for this core configuration.

    Example:
        ```python
        # Create core with all components
        core = Core(
            context=ContextCompose(
                ProfileContextSystem(...),
                KnowledgeContextSystem(...),
                context_template=Template("..."),
            ),
            completion=OpenAICompletion(...),
            toolset=[GetWeather(), SearchWeb()],
            model_id="assistant-v1@default",
        )

        await core.init()

        # Generate completion
        messages = Messages(messages=[
            Message(role="user", content="What's the weather in SF?")
        ])

        completion = await core.complete(
            messages,
            extras={"profile.info": {"location": "San Francisco"}}
        )

        print(completion.message.content)

        await core.close()
        ```

    Note:
        - All three components (context, completion, toolset) must be configured
        - Context enrichment happens before every completion
        - Toolset is passed to completion for automatic tool calling
        - model_id is used for completion metadata
    """

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
        """Initialize all core components.

        Initializes the completion system and all context systems in the
        composition. Must be called before generating completions.
        """
        await self.completion.init()
        await self.context.init()

    async def close(self) -> None:
        """Close and cleanup all core components.

        Closes the completion system and all context systems, releasing
        any held resources. Should be called when the core is no longer needed.
        """
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
        """Generate completion through the full PersonaX pipeline.

        Executes the complete flow:
        1. Enrich messages with context (ContextCompose.build)
        2. Generate completion with tools (CompletionSystem.complete)

        Args:
            messages: Input conversation messages.
            chatcmpl_id: Optional completion ID for response.
            stream: Whether to stream the response.
            max_completion_tokens: Maximum tokens to generate.
            prompt_cache_key: Optional key for prompt caching.
            extras: Additional context data (e.g., user profile info).

        Returns:
            Completion object if stream=False, AsyncStream if stream=True.

        Example:
            ```python
            # Non-streaming with context extras
            completion = await core.complete(
                messages,
                extras={
                    "profile.info": {
                        "prefname": "Alice",
                        "ip": "123.45.67.89"
                    }
                },
                max_completion_tokens=500,
            )

            # Streaming
            stream = await core.complete(messages, stream=True)
            async for chunk in stream:
                if chunk.delta.content:
                    print(chunk.delta.content, end="")
            ```
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
    """Base class for PersonaX persona implementations.

    PersonaX provides a high-level abstraction for AI personas, wrapping
    the Core pipeline with identity management. Each PersonaX instance
    represents a unique AI model configuration identified by name, version,
    and scenario.

    Class Variables:
        name: Base name of the persona (required, must be defined by subclasses).
        version: Version identifier. Defaults to "latest".
        scenario: Scenario or use case identifier. Defaults to "default".

    Properties:
        id: Unique identifier built from name, version, and scenario.

    Attributes:
        core: The underlying Core instance managing the completion pipeline.

    Example:
        ```python
        class Assistant(PersonaX):
            name = "assistant"
            version = "v1"
            scenario = "general"

            def __init__(self, core: Core):
                super().__init__(core)


        # Create persona
        persona = Assistant(core)
        print(persona.id)  # "assistant-v1@general"

        await persona.init()

        # Generate completion
        completion = await persona.complete(messages)

        await persona.close()


        # Multiple scenarios
        class AssistantS2S(PersonaX):
            name = "assistant"
            version = "v1"
            scenario = "s2s"

        class AssistantInquiry(PersonaX):
            name = "assistant"
            version = "v1"
            scenario = "inquiry"

        # Each has unique ID for orchestration
        print(AssistantS2S(core).id)  # "assistant-v1@s2s"
        print(AssistantInquiry(core).id)  # "assistant-v1@inquiry"
        ```

    Note:
        - Subclasses must define the `name` class variable
        - PersonaX instances are hashable by their ID
        - IDs are used for persona registration in orchestration systems
        - Multiple personas can share the same name with different versions/scenarios
    """

    core: Core

    name: t.ClassVar[str]
    version: t.ClassVar[t.Literal["latest"] | str] = "latest"
    scenario: t.ClassVar[t.Literal["default"] | str] = "default"

    @classproperty
    def id(self) -> str:
        """Get the unique identifier for this persona.

        Returns:
            Formatted ID string combining name, version, and scenario.
        """
        return build_id(self.name, self.version, self.scenario)

    def __init_subclass__(cls, **kwargs: t.Any) -> None:
        """Validate that subclasses define required class variables.

        Raises:
            NotImplementedError: If `name` is not defined or is not a string.
        """
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
        """Hash persona by ID for use in sets/dicts.

        Returns:
            Hash of the persona ID.
        """
        return hash(self.id)

    def __init__(self, core: Core) -> None:
        self.core = core

    async def init(self) -> None:
        """Initialize the underlying core components."""
        await self.core.init()

    async def close(self) -> None:
        """Close and cleanup the underlying core components."""
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
        """Generate completion using this persona.

        Delegates to the underlying core's complete method.

        Args:
            messages: Input conversation messages.
            chatcmpl_id: Optional completion ID for response.
            stream: Whether to stream the response.
            max_completion_tokens: Maximum tokens to generate.
            prompt_cache_key: Optional key for prompt caching.
            extras: Additional context data.

        Returns:
            Completion object if stream=False, AsyncStream if stream=True.
        """
        return await self.core.complete(
            messages=messages,
            chatcmpl_id=chatcmpl_id,
            stream=stream,
            max_completion_tokens=max_completion_tokens,
            prompt_cache_key=prompt_cache_key,
            extras=extras,
        )
