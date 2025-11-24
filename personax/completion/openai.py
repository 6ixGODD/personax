from __future__ import annotations

import json
import logging
import time
import typing as t

import openai
import openai.resources.chat as chat_rc
import openai.types.chat as chat_t
import openai.types.chat.chat_completion_message_function_tool_call_param as fn_param

from personax.completion import CompletionSystem
from personax.exceptions import ToolCallError
from personax.tools import BaseToolType
from personax.types.compat.message import Message
from personax.types.compat.message import Messages
from personax.types.compat.tool_calls import Function
from personax.types.compat.tool_calls import ToolCalls
from personax.types.compat.tool_calls import ToolCallsParams
from personax.types.completion import Completion
from personax.types.completion import CompletionMessage
from personax.types.completion_chunk import CompletionChunk
from personax.types.completion_chunk import CompletionChunkDelta
from personax.types.stream import AsyncStream
from personax.types.usage import Usage
from personax.utils import UNSET
from personax.utils import Unset
from personax.utils import filter_kwargs

logger = logging.getLogger("persomna.completion.openai")


def map_finish_reason(
    reason: t.Literal["stop", "length", "tool_calls", "content_filter", "function_call"] | None,
) -> t.Literal["stop", "length", "content_filter"]:
    """Map OpenAI finish reasons to standardized reasons.

    Args:
        reason: OpenAI finish reason.

    Returns:
        Standardized finish reason (tool_calls/function_call mapped to stop).
    """
    if reason and reason in ("stop", "length", "content_filter"):
        return t.cast(t.Literal["stop", "length", "content_filter"], reason)
    return "stop"


class OpenAIConfig(t.NamedTuple):
    """Configuration for OpenAI API client."""

    api_key: str
    """OpenAI API authentication key."""

    model: str
    """Model identifier for completions."""

    base_url: str
    """API endpoint base URL."""

    organization: str | None = None
    """Optional organization ID."""

    project: str | None = None
    """Optional project ID."""

    timeout: float | None = None
    """Request timeout in seconds."""

    max_retries: int = 3
    """Maximum number of retry attempts."""

    default_headers: t.Mapping[str, str] | None = None
    """Additional headers for all requests."""

    default_query: t.Mapping[str, object] | None = None
    """Additional query parameters for all requests."""


class OpenAICompletion(CompletionSystem):
    """OpenAI-compatible completion system with automatic tool calling.

    Implements the CompletionSystem interface for OpenAI-compatible APIs,
    with built-in support for multi-turn tool calling, streaming, and
    configurable generation parameters.

    Key Features:
    - Automatic tool call iteration until final response
    - Streaming and non-streaming modes
    - Configurable temperature, penalties, and other parameters
    - Support for prompt caching via prompt_cache_key
    - Comprehensive logging for debugging

    The system automatically handles tool calling loops:
    1. LLM generates response (potentially with tool calls)
    2. Tool calls are executed and results added to history
    3. Process repeats until LLM produces final response without tool calls

    Attributes:
        model: The OpenAI model identifier for completions.
        client: Configured AsyncOpenAI client instance.
        temperature: Sampling temperature (0-2).
        presence_penalty: Presence penalty (-2 to 2).
        frequency_penalty: Frequency penalty (-2 to 2).
        verbosity: Response verbosity level.
        top_p: Nucleus sampling parameter (0-1).

    Args:
        openai_config: OpenAI client configuration.
        temperature: Sampling temperature for randomness control.
        presence_penalty: Penalty for token presence (encourages new topics).
        frequency_penalty: Penalty for token frequency (reduces repetition).
        verbosity: Response detail level ("low", "medium", "high").
        top_p: Nucleus sampling threshold.

    Example:
        ```python
        # Basic setup
        completion_system = OpenAICompletion(
            openai_config=OpenAIConfig(
                api_key="sk-...",
                model="gpt-4",
                base_url="https://api.openai.com/v1",
                timeout=30.0,
                max_retries=3,
            ),
            temperature=0.7,
            presence_penalty=0.0,
            frequency_penalty=0.0,
        )


        # Non-streaming completion
        messages = Messages.from_raws(
            raws=Messages(
                messages=[
                    Message(role="user", content="What is 2+2?")
                ]
            ),
            sys_prompt="You are a helpful math assistant.",
        )

        completion = await completion_system.complete(
            messages,
            model="gpt-4",
            max_completion_tokens=100,
        )
        print(completion.message.content)  # "2+2 equals 4."


        # Streaming completion
        stream = await completion_system.complete(
            messages,
            model="gpt-4",
            stream=True,
        )
        async for chunk in stream:
            if chunk.delta.content:
                print(chunk.delta.content, end="", flush=True)


        # With automatic tool calling
        completion = await completion_system.complete(
            messages,
            model="gpt-4",
            tools=[GetWeather(), SearchWeb()],
            max_completion_tokens=500,
        )
        # System automatically iterates through tool calls


        # With prompt caching
        completion = await completion_system.complete(
            messages,
            model="gpt-4",
            prompt_cache_key="math_assistant_v1",  # Cache system prompt
        )
        ```

    Note:
        - The model parameter in complete() is for metadata only
        - Actual model is configured in OpenAIConfig during initialization
        - Tool calling triggers automatic multi-turn iteration
        - Streaming mode yields chunks for each iteration
        - All OpenAI errors are propagated to caller
    """

    def __init__(
        self,
        *,
        openai_config: OpenAIConfig,
        temperature: float | Unset = UNSET,
        presence_penalty: float | Unset = UNSET,
        frequency_penalty: float | Unset = UNSET,
        verbosity: t.Literal["low", "medium", "high"] | Unset = UNSET,
        top_p: float | Unset = UNSET,
    ):
        super().__init__()
        self.model = openai_config.model
        logger.debug(
            "Initializing OpenAICompletion with model=%s, base_url=%s",
            self.model,
            openai_config.base_url,
        )

        self.client = openai.AsyncOpenAI(
            api_key=openai_config.api_key,
            organization=openai_config.organization,
            project=openai_config.project,
            base_url=openai_config.base_url,
            timeout=openai_config.timeout,
            max_retries=openai_config.max_retries,
            default_headers=openai_config.default_headers,
            default_query=openai_config.default_query,
        )
        self.temperature = openai.omit if isinstance(temperature, Unset) else temperature
        self.presence_penalty = (
            openai.omit if isinstance(presence_penalty, Unset) else presence_penalty
        )
        self.frequency_penalty = (
            openai.omit if isinstance(frequency_penalty, Unset) else frequency_penalty
        )
        self.verbosity = openai.omit if isinstance(verbosity, Unset) else verbosity
        self.top_p = openai.omit if isinstance(top_p, Unset) else top_p

        logger.debug(
            "OpenAI client initialized with timeout=%s, max_retries=%s",
            openai_config.timeout,
            openai_config.max_retries,
        )

    async def complete(
        self,
        messages: Messages,
        *,
        tools: t.Sequence[BaseToolType] = (),
        chatcmpl_id: str | Unset = UNSET,
        stream: bool = False,
        max_completion_tokens: int | Unset = UNSET,
        model: str,
        prompt_cache_key: str | Unset = UNSET,
        **kwargs: t.Any,
    ) -> Completion | AsyncStream[CompletionChunk]:
        """Generate completion with optional tool calling.

        Dispatches to streaming or non-streaming implementation based on
        the stream parameter. Handles automatic tool call iteration in
        both modes.

        Args:
            messages: Input messages with system prompt.
            tools: Tools available for function calling.
            chatcmpl_id: Optional completion ID for response.
            stream: Whether to stream the response.
            max_completion_tokens: Maximum tokens to generate.
            model: Model identifier for response metadata.
            prompt_cache_key: Optional key for prompt caching.
            **kwargs: Additional OpenAI API parameters.

        Returns:
            Completion object if stream=False, AsyncStream if stream=True.
        """
        msg_count = len(list(messages))
        tool_names = [tool.__function_name__ for tool in tools]

        logger.debug(
            "Starting completion: stream=%s, model=%s, messages_count=%s, tools=%s",
            stream,
            model,
            msg_count,
            tool_names,
        )

        if stream:
            logger.debug("Routing to streaming completion")
            return await self._stream_complete(
                messages=messages.messages,
                tools=tools,
                chatcmpl_id=chatcmpl_id,
                max_completion_tokens=max_completion_tokens,
                model=model,
                prompt_cache_key=prompt_cache_key,
                **kwargs,
            )
        logger.debug("Routing to sync completion")
        return await self._sync_complete(
            messages=messages.messages,
            tools=tools,
            chatcmpl_id=chatcmpl_id,
            max_completion_tokens=max_completion_tokens,
            model=model,
            prompt_cache_key=prompt_cache_key,
            **kwargs,
        )

    async def _sync_complete(
        self,
        messages: t.Iterable[Message],
        *,
        tools: t.Sequence[BaseToolType] = (),
        chatcmpl_id: str | Unset = UNSET,
        max_completion_tokens: int | Unset = UNSET,
        model: str,
        prompt_cache_key: str | Unset = UNSET,
        **kwargs: t.Any,
    ) -> Completion:
        """Non-streaming completion with automatic tool call iteration.

        Implements the full tool calling loop:
        1. Send messages to LLM
        2. If response contains tool calls, execute them
        3. Add tool results to message history
        4. Repeat until LLM produces final response

        Args:
            messages: Input message sequence.
            tools: Available tool instances.
            chatcmpl_id: Optional completion ID.
            max_completion_tokens: Token generation limit.
            model: Model identifier for metadata.
            prompt_cache_key: Optional cache key.
            **kwargs: Additional API parameters.

        Returns:
            Final completion after all tool iterations.
        """
        msgs = list(messages)  # type: t.List[Message | ToolCalls | ToolCallsParams]
        tools_map = {tool.__function_name__: tool for tool in tools}
        iteration = 0

        logger.debug("Sync completion started with %s initial messages", len(msgs))
        logger.debug("Available tools: %s", list(tools_map.keys()))

        while True:
            iteration += 1
            logger.debug("--- Iteration %s ---", iteration)

            created = int(time.time())
            message_list = self._build_msgs(msgs)
            tool_list = (
                [t.cast(chat_t.ChatCompletionFunctionToolParam, tool.schema_dict) for tool in tools]
                if tools
                else openai.omit
            )

            logger.debug("Built %s OpenAI messages", len(message_list))
            if tools:
                logger.debug("Sending %s tool schemas", len(tools))

            # Log message structure for debugging
            for i, msg in enumerate(message_list):
                logger.debug(
                    "Message %s: role=%s, has_content=%s, has_tool_calls=%s",
                    i,
                    msg.get("role"),
                    "content" in msg,
                    "tool_calls" in msg,
                )

            completion = await self.client.chat.completions.create(
                messages=message_list,
                model=self.model,
                stream=False,
                tools=tool_list,
                max_completion_tokens=openai.omit
                if isinstance(max_completion_tokens, Unset)
                else max_completion_tokens,
                temperature=self.temperature,
                presence_penalty=self.presence_penalty,
                frequency_penalty=self.frequency_penalty,
                verbosity=self.verbosity,
                prompt_cache_key=openai.omit
                if isinstance(prompt_cache_key, Unset)
                else prompt_cache_key,
                parallel_tool_calls=True if tools else openai.omit,
                top_p=self.top_p,
                **filter_kwargs(chat_rc.AsyncCompletions.create, kwargs),
            )  # type: chat_t.ChatCompletion

            choice = completion.choices[0]

            logger.debug(
                "OpenAI response: finish_reason=%s, content_length=%s, tool_calls_count=%s",
                choice.finish_reason,
                len(choice.message.content or ""),
                len(choice.message.tool_calls or []),
            )

            # Add assistant message to history
            assistant_msg = Message(role="assistant", content=choice.message.content or "")
            msgs.append(assistant_msg)
            logger.debug(
                "Added assistant message to history (length: %s)", len(assistant_msg.content)
            )

            # Check for tool calls
            if choice.message.tool_calls:
                logger.debug("Processing %s tool calls", len(choice.message.tool_calls))

                for i, tool_call in enumerate(choice.message.tool_calls):
                    if not isinstance(tool_call, chat_t.ChatCompletionMessageFunctionToolCall):
                        logger.debug("Skipping non-function tool call at index %s", i)
                        continue
                    func_name = tool_call.function.name
                    call_id = tool_call.id

                    logger.debug(
                        "Tool call %s: %s(%s...)",
                        i + 1,
                        func_name,
                        tool_call.function.arguments[:100],
                    )
                    tool_params = ToolCallsParams(
                        call_id=call_id,
                        function=Function(name=func_name, arguments=tool_call.function.arguments),
                    )
                    msgs.append(tool_params)
                    logger.debug("Added tool call params to history: %s", call_id)

                    if func_name in tools_map:
                        logger.debug("Executing tool: %s", func_name)
                        args = json.loads(tool_call.function.arguments)  # type: dict[str, object]
                        logger.debug("Tool args: %s", args)

                        start_time = time.time()
                        try:
                            result = tools_map[func_name](**args)
                            if isinstance(result, t.Awaitable):
                                result = await result
                        except ToolCallError as e:
                            logger.error("Error executing tool %s: %s", func_name, str(e))
                            result = f"Error executing tool {func_name}"
                        exec_time = time.time() - start_time

                        logger.debug("Tool %s executed in %.3fs", func_name, exec_time)
                        logger.debug(
                            "Tool result type: %s, preview: %s", type(result), str(result)[:200]
                        )

                        # Add tool result
                        result_str: str | list[str]
                        if isinstance(result, str):
                            result_str = result
                        elif isinstance(result, list):
                            result_str = [
                                json.dumps(r) if not isinstance(r, str) else r for r in result
                            ]
                        else:
                            result_str = json.dumps(result)

                        tool_result = ToolCalls(call_id=call_id, content=result_str)
                        msgs.append(tool_result)
                        logger.debug("Added tool result to history: %s", call_id)
                    else:
                        logger.debug("Tool %s not found in available tools", func_name)
                        error_result = ToolCalls(
                            call_id=call_id, content=f"Tool {func_name} not available"
                        )
                        msgs.append(error_result)

                logger.debug("Continuing to next iteration with %s total messages", len(msgs))
                continue  # Continue the loop for next iteration

            # No tool calls, return final result
            logger.debug("No tool calls found, preparing final completion")

            usage_info = None
            if completion.usage:
                usage_info = Usage(
                    completion_tokens=completion.usage.completion_tokens,
                    prompt_tokens=completion.usage.prompt_tokens,
                    total_tokens=completion.usage.total_tokens,
                )
                logger.debug(
                    "Usage: %s + %s = %s tokens",
                    usage_info.prompt_tokens,
                    usage_info.completion_tokens,
                    usage_info.total_tokens,
                )

            final_completion = Completion(
                id=completion.id if isinstance(chatcmpl_id, Unset) else chatcmpl_id,
                message=CompletionMessage(
                    content=choice.message.content, refusal=choice.message.refusal, reason=None
                ),
                finish_reason=map_finish_reason(choice.finish_reason),
                created=created,
                model=model,
                usage=usage_info,
            )

            logger.debug("Sync completion finished after %s iterations", iteration)
            return final_completion
        raise RuntimeError("Unreachable code reached in `_sync_complete`")  # for mypy

    async def _stream_complete(
        self,
        messages: t.Iterable[Message],
        *,
        tools: t.Sequence[BaseToolType] = (),
        chatcmpl_id: str | Unset = UNSET,
        max_completion_tokens: int | Unset = UNSET,
        model: str,
        prompt_cache_key: str | Unset = UNSET,
        **kwargs: t.Any,
    ) -> AsyncStream[CompletionChunk]:
        """Streaming completion with automatic tool call iteration.

        Streams chunks from each iteration of the tool calling loop. Tool
        calls are assembled from delta chunks, executed, and results added
        to history before starting the next iteration.

        Args:
            messages: Input message sequence.
            tools: Available tool instances.
            chatcmpl_id: Optional completion ID.
            max_completion_tokens: Token generation limit.
            model: Model identifier for metadata.
            prompt_cache_key: Optional cache key.
            **kwargs: Additional API parameters.

        Returns:
            AsyncStream yielding chunks for all iterations.
        """
        msg_list = list(messages)  # type: t.List[Message | ToolCalls | ToolCallsParams]
        tools_map = {tool.__function_name__: tool for tool in tools}

        logger.debug("Stream completion started with %s initial messages", len(msg_list))
        logger.debug("Available tools: %s", list(tools_map.keys()))

        async def _stream_gen() -> t.AsyncGenerator[CompletionChunk, None]:
            nonlocal msg_list
            iteration = 0

            while True:
                iteration += 1
                logger.debug("--- Stream Iteration %s ---", iteration)

                created = int(time.time())
                message_list = self._build_msgs(msg_list)
                tool_list = (
                    [
                        t.cast(chat_t.ChatCompletionFunctionToolParam, tool.schema_dict)
                        for tool in tools
                    ]
                    if tools
                    else openai.omit
                )
                logger.debug("Built %s OpenAI messages for streaming", len(message_list))

                completion = await self.client.chat.completions.create(
                    messages=message_list,
                    model=self.model,
                    stream=True,
                    tools=tool_list,
                    max_completion_tokens=openai.omit
                    if isinstance(max_completion_tokens, Unset)
                    else max_completion_tokens,
                    temperature=self.temperature,
                    presence_penalty=self.presence_penalty,
                    frequency_penalty=self.frequency_penalty,
                    verbosity=self.verbosity,
                    prompt_cache_key=openai.omit
                    if isinstance(prompt_cache_key, Unset)
                    else prompt_cache_key,
                    parallel_tool_calls=True if tools else openai.omit,
                    top_p=self.top_p,
                    **filter_kwargs(chat_rc.AsyncCompletions.create, kwargs),
                )

                content_buffer = ""
                tool_calls_buffer = {}
                has_tool_calls = False
                chunk_count = 0

                logger.debug("Starting to process stream chunks")

                async for chunk in completion:
                    chunk_count += 1
                    choice = chunk.choices[0] if chunk.choices else None
                    if not choice:
                        logger.debug("Chunk %s: no choice data", chunk_count)
                        continue

                    delta = choice.delta
                    logger.debug(
                        "Chunk %s: has_content=%s, has_tool_calls=%s, finish_reason=%s",
                        chunk_count,
                        bool(delta.content),
                        bool(delta.tool_calls),
                        choice.finish_reason,
                    )

                    # Handle content
                    if delta.content:
                        content_buffer += delta.content
                        logger.debug(
                            "Content chunk: '%s' (total: %s chars)",
                            delta.content,
                            len(content_buffer),
                        )

                        yield CompletionChunk(
                            id=chunk.id if isinstance(chatcmpl_id, Unset) else chatcmpl_id,
                            delta=CompletionChunkDelta(content=delta.content),
                            finish_reason=(
                                map_finish_reason(choice.finish_reason)
                                if choice.finish_reason
                                else None
                            ),
                            created=created,
                            model=model,
                            usage=Usage(
                                completion_tokens=chunk.usage.completion_tokens,
                                prompt_tokens=chunk.usage.prompt_tokens,
                                total_tokens=chunk.usage.total_tokens,
                            )
                            if chunk.usage
                            else None,
                        )

                    # Handle tool calls
                    if delta.tool_calls:
                        has_tool_calls = True
                        logger.debug("Processing tool call deltas: %s items", len(delta.tool_calls))

                        for tool_call in delta.tool_calls:
                            idx = tool_call.index
                            func = tool_call.function

                            logger.debug(
                                "Tool call delta %s: id=%s, name=%s, args_chunk='%s'",
                                idx,
                                tool_call.id,
                                func.name if func else None,
                                func.arguments if func else None,
                            )

                            if idx not in tool_calls_buffer:
                                tool_calls_buffer[idx] = {
                                    "id": tool_call.id or "",
                                    "function": {"name": "", "arguments": ""},
                                }
                                logger.debug("Created new tool call buffer for index %s", idx)

                            if tool_call.function:
                                if tool_call.function.name:
                                    tool_calls_buffer[idx]["function"]["name"] += (
                                        tool_call.function.name
                                    )  # type: ignore
                                if tool_call.function.arguments:
                                    tool_calls_buffer[idx]["function"]["arguments"] += (
                                        tool_call.function.arguments
                                    )  # type: ignore

                    # Handle finish
                    if choice.finish_reason:
                        logger.debug("Stream finished with reason: %s", choice.finish_reason)

                        yield CompletionChunk(
                            id=chunk.id if isinstance(chatcmpl_id, Unset) else chatcmpl_id,
                            delta=CompletionChunkDelta(),
                            finish_reason=map_finish_reason(choice.finish_reason),
                            created=created,
                            model=model,
                            usage=Usage(
                                completion_tokens=chunk.usage.completion_tokens,
                                prompt_tokens=chunk.usage.prompt_tokens,
                                total_tokens=chunk.usage.total_tokens,
                            )
                            if chunk.usage
                            else None,
                        )
                        break

                logger.debug(
                    "Processed %s chunks, content_length=%s", chunk_count, len(content_buffer)
                )

                # Add assistant message to history
                assistant_msg = Message(role="assistant", content=content_buffer)
                msg_list.append(assistant_msg)
                logger.debug("Added assistant message to history (stream)")

                # Process tool calls if any
                if has_tool_calls and tool_calls_buffer:
                    logger.debug("Processing %s tool calls from stream", len(tool_calls_buffer))

                    for idx, tool_call_data in tool_calls_buffer.items():
                        func_name: str = tool_call_data["function"]["name"]  # type: ignore
                        call_id: str = tool_call_data["id"]  # type: ignore
                        args_str: str = tool_call_data["function"]["arguments"]  # type: ignore

                        logger.debug(
                            "Stream tool call %s: %s(%s...)", idx, func_name, args_str[:100]
                        )

                        # Add tool call params to history
                        tool_params = ToolCallsParams(
                            call_id=call_id,
                            function=Function(name=func_name, arguments=args_str),
                        )
                        msg_list.append(tool_params)
                        logger.debug("Added stream tool call params: %s", call_id)

                        # Execute tool and add result
                        if func_name in tools_map:
                            logger.debug("Executing stream tool: %s", func_name)
                            args = json.loads(args_str)

                            start_time = time.time()
                            try:
                                result = tools_map[func_name](**args)
                                if isinstance(result, t.Awaitable):
                                    result = await result
                            except ToolCallError as e:
                                logger.error("Error executing tool %s: %s", func_name, str(e))
                                result = f"Error executing tool {func_name}"
                            exec_time = time.time() - start_time

                            logger.debug("Stream tool %s executed in %.3fs", func_name, exec_time)

                            result_str: str | list[str]
                            if isinstance(result, str):
                                result_str = result
                            elif isinstance(result, list):
                                result_str = [
                                    json.dumps(r) if not isinstance(r, str) else r for r in result
                                ]
                            else:
                                result_str = json.dumps(result)

                            tool_result = ToolCalls(call_id=call_id, content=result_str)
                            msg_list.append(tool_result)
                            logger.debug("Added stream tool result: %s", call_id)
                        else:
                            logger.debug("Stream tool %s not found in available tools", func_name)
                            error_result = ToolCalls(
                                call_id=call_id, content=f"Tool {func_name} not available"
                            )
                            msg_list.append(error_result)

                    logger.debug(
                        "Continuing stream to next iteration with %s total messages", len(msg_list)
                    )
                    continue  # Continue the loop for next iteration

                # No tool calls, we're done
                logger.debug("Stream completion finished after %s iterations", iteration)
                break

        return AsyncStream(_stream_gen())

    @staticmethod
    def _build_msgs(
        messages: list[Message | ToolCalls | ToolCallsParams],
    ) -> list[chat_t.ChatCompletionMessageParam]:
        """Build OpenAI message list from Message objects."""
        logger.debug("Building OpenAI messages from %s input messages", len(messages))

        msgs = []  # type: t.List[chat_t.ChatCompletionMessageParam]
        msg_types = [type(msg).__name__ for msg in messages]
        logger.debug("Input message types: %s", msg_types)

        for i, msg in enumerate(messages):
            if isinstance(msg, Message):
                if msg.role == "system":
                    msgs.append(
                        chat_t.ChatCompletionSystemMessageParam(role="system", content=msg.content)
                    )
                    logger.debug("Built system message %s: %s chars", i, len(msg.content))
                elif msg.role == "user":
                    msgs.append(
                        chat_t.ChatCompletionUserMessageParam(role="user", content=msg.content)
                    )
                    logger.debug("Built user message %s: %s chars", i, len(msg.content))
                elif msg.role == "assistant":
                    msgs.append(
                        chat_t.ChatCompletionAssistantMessageParam(
                            role="assistant", content=msg.content
                        )
                    )
                    logger.debug("Built assistant message %s: %s chars", i, len(msg.content))
            elif isinstance(msg, ToolCallsParams):
                logger.debug("Processing tool call params %s: %s", i, msg.function.name)

                # Look for existing assistant message to add tool calls to
                if msgs and msgs[-1].get("role") == "assistant":
                    # Add tool calls to the last assistant message
                    if "tool_calls" not in msgs[-1]:
                        msgs[-1]["tool_calls"] = []  # type: ignore
                    msgs[-1]["tool_calls"].append(  # type: ignore
                        chat_t.ChatCompletionMessageFunctionToolCallParam(
                            id=msg.call_id,
                            function=fn_param.Function(
                                name=msg.function.name, arguments=msg.function.arguments
                            ),
                            type="function",
                        )
                    )
                    logger.debug(
                        "Added tool call to existing assistant message: %s", msg.function.name
                    )
                else:
                    # Create new assistant message with tool calls
                    msgs.append(
                        chat_t.ChatCompletionAssistantMessageParam(
                            role="assistant",
                            tool_calls=[
                                chat_t.ChatCompletionMessageFunctionToolCallParam(
                                    id=msg.call_id,
                                    function=fn_param.Function(
                                        name=msg.function.name, arguments=msg.function.arguments
                                    ),
                                    type="function",
                                )
                            ],
                        )
                    )
                    logger.debug(
                        "Created new assistant message with tool call: %s", msg.function.name
                    )
            elif isinstance(msg, ToolCalls):
                msgs.append(
                    chat_t.ChatCompletionToolMessageParam(
                        role="tool", content=str(msg.content), tool_call_id=msg.call_id
                    )
                )
                content_preview = (
                    str(msg.content)[:100] if isinstance(msg.content, str) else str(msg.content)
                )
                logger.debug("Built tool result message %s: %s...", i, content_preview)

        logger.debug("Built %s OpenAI messages", len(msgs))
        return msgs

    @staticmethod
    async def _parse_stream(
        completion: t.AsyncIterable[chat_t.ChatCompletionChunk],
        chatcmpl_id: str,
        created: int,
        model: str,
    ) -> AsyncStream[CompletionChunk]:
        """Parse OpenAI stream chunks into standardized
        CompletionChunks."""
        logger.debug("Parsing stream with chatcmpl_id=%s, model=%s", chatcmpl_id, model)

        async def _gen() -> t.AsyncGenerator[CompletionChunk, None]:
            chunk_count = 0
            async for chunk in completion:
                chunk_count += 1
                choice = chunk.choices[0] if chunk.choices else None
                if not choice:
                    logger.debug("Stream chunk %s: no choice data", chunk_count)
                    continue

                delta = choice.delta
                logger.debug(
                    "Stream chunk %s: content=%s, finish_reason=%s",
                    chunk_count,
                    bool(delta.content),
                    choice.finish_reason,
                )

                yield CompletionChunk(
                    id=chatcmpl_id,
                    delta=CompletionChunkDelta(
                        content=delta.content, refusal=delta.refusal, reason=None
                    ),
                    finish_reason=(
                        map_finish_reason(choice.finish_reason) if choice.finish_reason else None
                    ),
                    created=created,
                    model=model,
                    usage=Usage(
                        completion_tokens=chunk.usage.completion_tokens,
                        prompt_tokens=chunk.usage.prompt_tokens,
                        total_tokens=chunk.usage.total_tokens,
                    )
                    if chunk.usage
                    else None,
                )
            logger.debug("Stream parsing completed after %s chunks", chunk_count)

        return AsyncStream(_gen())
