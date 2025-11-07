from __future__ import annotations

import abc
import typing as t

from personax.tools import BaseToolType
from personax.types.compat.message import Messages
from personax.types.completion import Completion
from personax.types.completion_chunk import CompletionChunk
from personax.types.stream import AsyncStream
from personax.utils import UNSET
from personax.utils import AsyncContextMixin
from personax.utils import Unset


class CompletionSystem(AsyncContextMixin, abc.ABC):
    """Abstract base class for LLM text completion systems.

    CompletionSystem defines the interface for language model completion
    implementations. It handles text generation with support for both
    streaming and non-streaming responses, tool calling capabilities,
    and flexible configuration.

    The system is designed to be stateless and reusable across multiple
    completion requests, with optional lifecycle management via init/close
    for implementations that require resource setup (e.g., model loading,
    connection pooling).

    Key Features:
    - Unified interface for different LLM providers (OpenAI, Anthropic, etc.)
    - Streaming and non-streaming completion modes
    - Tool/function calling support with automatic iteration
    - Prompt caching support for providers that offer KV cache
    - Flexible parameter passing for provider-specific options

    Example:
        ```python
        # Basic usage
        completion_system = OpenAICompletion(
            openai_config=OpenAIConfig(
                api_key="sk-...",
                model="gpt-4",
                base_url="https://api.openai.com/v1",
            )
        )

        await completion_system.init()

        messages = Messages(
            messages=[
                Message(
                    role="system",
                    content="You are a helpful assistant.",
                ),
                Message(
                    role="user",
                    content="What is the capital of France?",
                ),
            ]
        )

        # Non-streaming completion
        completion = await completion_system.complete(
            messages,
            model="gpt-4",
            max_completion_tokens=100,
        )
        print(completion.message.content)  # "Paris"


        # Streaming completion
        stream = await completion_system.complete(
            messages,
            model="gpt-4",
            stream=True,
        )
        async for chunk in stream:
            if chunk.delta.content:
                print(chunk.delta.content, end="", flush=True)


        # With tools
        from personax.tools import BaseTool


        class GetWeather(BaseTool):
            __function_name__ = "get_weather"
            __function_description__ = "Get current weather"

            async def __call__(self, location: str) -> str:
                return f"Weather in {location}: Sunny, 72°F"


        completion = await completion_system.complete(
            messages,
            model="gpt-4",
            tools=[GetWeather()],
        )
        # System automatically handles tool calls and iterations

        await completion_system.close()
        ```
    """

    async def init(self) -> None:
        """Initialize completion system resources.

        Optional hook for setting up resources required by the completion
        system. Examples include:
        - Loading model weights for local deployment
        - Establishing connection pools
        - Initializing API clients
        - Setting up caches

        Default implementation does nothing. Override if needed.
        """

    async def close(self) -> None:
        """Clean up completion system resources.

        Optional hook for releasing resources when the completion system
        is no longer needed. Examples include:
        - Unloading model weights
        - Closing connection pools
        - Cleaning up temporary files
        - Flushing caches

        Default implementation does nothing. Override if needed.
        """

    @abc.abstractmethod
    async def complete(
        self,
        messages: Messages,
        *,
        tools: t.Sequence[BaseToolType] = (),
        chatcmpl_id: str | Unset = UNSET,
        stream: bool = False,
        max_completion_tokens: int | Unset = UNSET,
        model: str,
        _prompt_cache_key: str | Unset = UNSET,
        **kwargs: t.Any,
    ) -> Completion | AsyncStream[CompletionChunk]:
        """Generate LLM completion for the given messages.

        Performs text completion using the underlying language model. Supports
        both streaming and non-streaming modes, automatic tool calling with
        multi-turn iteration, and various configuration options.

        Args:
            messages: Conversation messages to complete. Must be a Messages
                object with system prompt and conversation history.
            tools: Sequence of tools available for the model to call. If
                provided, the system will automatically handle tool call
                iterations until the model produces a final response.
            chatcmpl_id: Optional completion ID to include in the response.
                If UNSET, the implementation will generate or use a
                provider-assigned ID. This parameter is for response metadata
                only and does not affect model behavior.
            stream: Whether to stream the response. If True, returns
                AsyncStream[CompletionChunk] for incremental output. If False,
                returns Completion with the full response.
            max_completion_tokens: Maximum number of tokens to generate in
                the completion. If UNSET, uses provider default or model limit.
            model: Model identifier for response metadata. This does NOT
                control which model is used for completion - the model should
                be configured during CompletionSystem initialization. This
                parameter only populates the `model` field in returned
                Completion/CompletionChunk objects.
            _prompt_cache_key: Optional cache key for prompt caching. Used by
                providers that support KV cache (e.g., prompt caching in
                Anthropic Claude). If UNSET, caching is not used. The key
                should uniquely identify the prompt prefix to cache.
            **kwargs: Additional provider-specific parameters. These are passed
                through to the underlying API client and should match the
                provider's API specification.

        Returns:
            Completion object containing the full response if stream=False, or
            AsyncStream[CompletionChunk] for incremental streaming if
            stream=True.

        Example:
            ```python
            # Non-streaming
            completion = await system.complete(
                messages,
                model="gpt-4",
                max_completion_tokens=500,
                temperature=0.7,  # Provider-specific kwarg
            )
            print(completion.message.content)


            # Streaming
            stream = await system.complete(
                messages,
                model="gpt-4",
                stream=True,
            )
            async for chunk in stream:
                if chunk.delta.content:
                    print(chunk.delta.content, end="")


            # With tools (automatic multi-turn handling)
            completion = await system.complete(
                messages,
                model="gpt-4",
                tools=[weather_tool, search_tool],
            )
            # System iterates: LLM -> tool call -> LLM -> ... -> final response


            # With prompt caching
            completion = await system.complete(
                messages,
                model="gpt-4",
                _prompt_cache_key="system_prompt_v1",  # Cache system messages
            )
            ```

        Note:
            - The `model` parameter is for metadata only. Configure the actual
              model during CompletionSystem initialization.
            - Tool calling may trigger multiple LLM calls in a loop until
              the model produces a final response without tool calls.
            - Streaming with tools will stream each iteration's response.
            - Provider-specific kwargs are filtered based on the underlying
              API client's signature to avoid errors.
        """
