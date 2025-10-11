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
from personax.tools import BaseTool
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
from personax.utils import filter_kwargs
from personax.utils import UNSET
from personax.utils import Unset

logger = logging.getLogger("persomna.completion.openai")


def map_finish_reason(
    reason: t.Literal["stop", "length", "tool_calls", "content_filter", "function_call"] | None
) -> t.Literal["stop", "length", "content_filter"]:
    if reason and reason in ("stop", "length", "content_filter"):
        return t.cast(t.Literal["stop", "length", "content_filter"], reason)
    return "stop"


class OpenAICompletion(CompletionSystem):

    def __init__(
        self,
        *,
        api_key: str,
        organization: str | None = None,
        project: str | None = None,
        model: str,
        base_url: str,
        timeout: float | None = None,
        max_retries: int = 3,
        default_headers: t.Mapping[str, str] | None = None,
        default_query: t.Mapping[str, object] | None = None,
        temperature: float | Unset = UNSET,
        presence_penalty: float | Unset = UNSET,
        frequency_penalty: float | Unset = UNSET,
        verbosity: t.Literal["low", "medium", "high"] | Unset = UNSET,
        top_p: float | Unset = UNSET,
    ):
        self.model = model
        logger.debug(f"Initializing OpenAICompletion with model={model}, base_url="
                     f"{base_url}")

        self.client = openai.AsyncOpenAI(api_key=api_key,
                                         organization=organization,
                                         project=project,
                                         base_url=base_url,
                                         timeout=timeout,
                                         max_retries=max_retries,
                                         default_headers=default_headers,
                                         default_query=default_query)
        self.temperature = temperature or openai.NOT_GIVEN
        self.presence_penalty = presence_penalty or openai.NOT_GIVEN
        self.frequency_penalty = frequency_penalty or openai.NOT_GIVEN
        self.verbosity = verbosity or openai.NOT_GIVEN
        self.top_p = top_p or openai.NOT_GIVEN

        logger.debug(f"OpenAI client initialized with timeout={timeout}, max_retries="
                     f"{max_retries}")

    async def complete(self,
                       messages: Messages,
                       *,
                       tools: t.Sequence[BaseTool] = (),
                       chatcmpl_id: str | Unset = UNSET,
                       stream: bool = False,
                       max_completion_tokens: int | Unset = UNSET,
                       model: str,
                       _prompt_cache_key: str | Unset = UNSET,
                       **kwargs: t.Any) -> Completion | AsyncStream[CompletionChunk]:
        msg_count = len(list(messages))
        tool_names = [tool.__function_name__ for tool in tools]

        logger.debug(f"Starting completion: stream={stream}, model={model}, "
                     f"messages_count={msg_count}, tools={tool_names}")

        if stream:
            logger.debug("Routing to streaming completion")
            return await self._stream_complete(messages=messages.messages,
                                               tools=tools,
                                               chatcmpl_id=chatcmpl_id,
                                               max_completion_tokens=max_completion_tokens,
                                               model=model,
                                               prompt_cache_key=_prompt_cache_key,
                                               **kwargs)
        else:
            logger.debug("Routing to sync completion")
            return await self._sync_complete(messages=messages.messages,
                                             tools=tools,
                                             chatcmpl_id=chatcmpl_id,
                                             max_completion_tokens=max_completion_tokens,
                                             model=model,
                                             prompt_cache_key=_prompt_cache_key,
                                             **kwargs)

    async def _sync_complete(self,
                             messages: t.Iterable[Message],
                             *,
                             tools: t.Sequence[BaseTool] = (),
                             chatcmpl_id: str | Unset = UNSET,
                             max_completion_tokens: int | Unset = UNSET,
                             model: str,
                             prompt_cache_key: str | Unset = UNSET,
                             **kwargs: t.Any) -> Completion:
        msg_list: t.List[Message | ToolCalls | ToolCallsParams] = list(messages)
        tools_map = {tool.__function_name__: tool for tool in tools}
        iteration = 0

        logger.debug(f"Sync completion started with {len(msg_list)} initial messages")
        logger.debug(f"Available tools: {list(tools_map.keys())}")

        while True:
            iteration += 1
            logger.debug(f"--- Iteration {iteration} ---")

            created = int(time.time())
            message_list = self._build_msgs(msg_list)
            tool_list = ([tool.schema_dict for tool in tools] if tools else openai.NOT_GIVEN)

            logger.debug(f"Built {len(message_list)} OpenAI messages")
            if tools:
                logger.debug(f"Sending {len(tools)} tool schemas")

            # Log message structure for debugging
            for i, msg in enumerate(message_list):
                logger.debug(f"Message {i}: role={msg.get('role')}, "
                             f"has_content={'content' in msg}, "
                             f"has_tool_calls={'tool_calls' in msg}")

            completion = await self.client.chat.completions.create(
                messages=message_list,
                model=self.model,
                stream=False,
                tools=tool_list,
                max_completion_tokens=max_completion_tokens or openai.NOT_GIVEN,
                temperature=self.temperature,
                presence_penalty=self.presence_penalty,
                frequency_penalty=self.frequency_penalty,
                verbosity=self.verbosity,
                prompt_cache_key=prompt_cache_key or openai.NOT_GIVEN,
                parallel_tool_calls=True if tools else openai.NOT_GIVEN,
                top_p=self.top_p,
                **filter_kwargs(chat_rc.AsyncCompletions.create, kwargs),
            )

            choice = completion.choices[0]
            msg = choice.message

            logger.debug(f"OpenAI response: finish_reason={choice.finish_reason}, "
                         f"content_length={len(msg.content or '')}, "
                         f"tool_calls_count={len(msg.tool_calls or [])}")

            # Add assistant message to history
            assistant_msg = Message(role="assistant", content=msg.content or "")
            msg_list.append(assistant_msg)
            logger.debug(f"Added assistant message to history (length: "
                         f"{len(assistant_msg.content)})")

            # Check for tool calls
            if msg.tool_calls:
                logger.debug(f"Processing {len(msg.tool_calls)} tool calls")

                for i, tool_call in enumerate(msg.tool_calls):
                    func_name = tool_call.function.name
                    call_id = tool_call.id
                    args_str = tool_call.function.arguments

                    logger.debug(f"Tool call {i + 1}: {func_name}({args_str[:100]}...)")

                    tool_params = ToolCallsParams(call_id=call_id,
                                                  function=Function(name=func_name,
                                                                    arguments=args_str))
                    msg_list.append(tool_params)
                    logger.debug(f"Added tool call params to history: {call_id}")

                    if func_name in tools_map:
                        logger.debug(f"Executing tool: {func_name}")
                        args: dict[str, object] = json.loads(args_str)
                        logger.debug(f"Tool args: {args}")

                        start_time = time.time()
                        result = tools_map[func_name](**args)
                        exec_time = time.time() - start_time

                        logger.debug(f"Tool {func_name} executed in {exec_time:.3f}s")
                        logger.debug(f"Tool result type: {type(result)}, "
                                     f"preview: {str(result)[:200]}")

                        # Add tool result
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
                        logger.debug(f"Added tool result to history: {call_id}")
                    else:
                        logger.debug(f"Tool {func_name} not found in available tools")
                        error_result = ToolCalls(call_id=call_id,
                                                 content=f"Tool {func_name} not available")
                        msg_list.append(error_result)

                logger.debug(f"Continuing to next iteration with {len(msg_list)} total "
                             f"messages")
                continue  # Continue the loop for next iteration

            # No tool calls, return final result
            logger.debug("No tool calls found, preparing final completion")

            usage_info = None
            if completion.usage:
                usage_info = Usage(completion_tokens=completion.usage.completion_tokens,
                                   prompt_tokens=completion.usage.prompt_tokens,
                                   total_tokens=completion.usage.total_tokens)
                logger.debug(f"Usage: {usage_info.prompt_tokens} + "
                             f"{usage_info.completion_tokens} = "
                             f"{usage_info.total_tokens} tokens")

            final_completion = Completion(id=chatcmpl_id if chatcmpl_id != UNSET else completion.id,
                                          message=CompletionMessage(content=msg.content,
                                                                    refusal=msg.refusal,
                                                                    reason=None),
                                          finish_reason=map_finish_reason(choice.finish_reason),
                                          created=created,
                                          model=model,
                                          usage=usage_info)

            logger.debug(f"Sync completion finished after {iteration} iterations")
            return final_completion
        raise RuntimeError("Unreachable code reached in `_sync_complete`")  # for mypy

    async def _stream_complete(self,
                               messages: t.Iterable[Message],
                               *,
                               tools: t.Sequence[BaseTool] = (),
                               chatcmpl_id: str | Unset = UNSET,
                               max_completion_tokens: int | Unset = UNSET,
                               model: str,
                               prompt_cache_key: str | Unset = UNSET,
                               **kwargs: t.Any) -> AsyncStream[CompletionChunk]:
        msg_list: t.List[Message | ToolCalls | ToolCallsParams] = list(messages)
        tools_map = {tool.__function_name__: tool for tool in tools}

        logger.debug(f"Stream completion started with {len(msg_list)} initial messages")
        logger.debug(f"Available tools: {list(tools_map.keys())}")

        async def _stream_gen() -> t.AsyncGenerator[CompletionChunk, None]:
            nonlocal msg_list
            iteration = 0

            while True:
                iteration += 1
                logger.debug(f"--- Stream Iteration {iteration} ---")

                created = int(time.time())
                message_list = self._build_msgs(msg_list)
                tool_list = ([tool.schema_dict for tool in tools] if tools else openai.NOT_GIVEN)

                logger.debug(f"Built {len(message_list)} OpenAI messages for streaming")

                completion = await self.client.chat.completions.create(
                    messages=message_list,
                    model=self.model,
                    stream=True,
                    tools=tool_list,
                    max_completion_tokens=max_completion_tokens or openai.NOT_GIVEN,
                    temperature=self.temperature,
                    presence_penalty=self.presence_penalty,
                    frequency_penalty=self.frequency_penalty,
                    verbosity=self.verbosity,
                    prompt_cache_key=(prompt_cache_key or openai.NOT_GIVEN),
                    parallel_tool_calls=True if tools else openai.NOT_GIVEN,
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
                        logger.debug(f"Chunk {chunk_count}: no choice data")
                        continue

                    delta = choice.delta
                    logger.debug(f"Chunk {chunk_count}: has_content="
                                 f"{bool(delta.content)}, "
                                 f"has_tool_calls={bool(delta.tool_calls)}, "
                                 f"finish_reason={choice.finish_reason}")

                    # Handle content
                    if delta.content:
                        content_buffer += delta.content
                        logger.debug(f"Content chunk: '{delta.content}' (total: "
                                     f"{len(content_buffer)} chars)")

                        yield CompletionChunk(
                            id=chatcmpl_id if chatcmpl_id != UNSET else chunk.id,
                            delta=CompletionChunkDelta(content=delta.content),
                            finish_reason=(map_finish_reason(choice.finish_reason)
                                           if choice.finish_reason else None),
                            created=created,
                            model=model,
                            usage=Usage(completion_tokens=chunk.usage.completion_tokens,
                                        prompt_tokens=chunk.usage.prompt_tokens,
                                        total_tokens=chunk.usage.total_tokens)
                            if chunk.usage else None,
                        )

                    # Handle tool calls
                    if delta.tool_calls:
                        has_tool_calls = True
                        logger.debug(f"Processing tool call deltas: "
                                     f"{len(delta.tool_calls)} items")

                        for tool_call in delta.tool_calls:
                            idx = tool_call.index
                            func = tool_call.function
                            logger.debug(f"Tool call delta {idx}: id={tool_call.id}, "
                                         f"name="
                                         f"{func.name if func else None}, "
                                         f"args_chunk="
                                         f"'{func.arguments if func else None}'")

                            if idx not in tool_calls_buffer:
                                tool_calls_buffer[idx] = {
                                    "id": tool_call.id or "",
                                    "function": {
                                        "name": "",
                                        "arguments": ""
                                    }
                                }
                                logger.debug(f"Created new tool call buffer for index "
                                             f"{idx}")

                            if tool_call.function:
                                if tool_call.function.name:
                                    tool_calls_buffer[idx]["function"][
                                        "name"] += tool_call.function.name
                                if tool_call.function.arguments:
                                    tool_calls_buffer[idx]["function"][
                                        "arguments"] += tool_call.function.arguments

                    # Handle finish
                    if choice.finish_reason:
                        logger.debug(f"Stream finished with reason: "
                                     f"{choice.finish_reason}")

                        yield CompletionChunk(
                            id=chatcmpl_id if chatcmpl_id != UNSET else chunk.id,
                            delta=CompletionChunkDelta(),
                            finish_reason=map_finish_reason(choice.finish_reason),
                            created=created,
                            model=model,
                            usage=Usage(completion_tokens=chunk.usage.completion_tokens,
                                        prompt_tokens=chunk.usage.prompt_tokens,
                                        total_tokens=chunk.usage.total_tokens)
                            if chunk.usage else None,
                        )
                        break

                logger.debug(f"Processed {chunk_count} chunks, content_length="
                             f"{len(content_buffer)}")

                # Add assistant message to history
                assistant_msg = Message(role="assistant", content=content_buffer)
                msg_list.append(assistant_msg)
                logger.debug(f"Added assistant message to history (stream)")

                # Process tool calls if any
                if has_tool_calls and tool_calls_buffer:
                    logger.debug(f"Processing {len(tool_calls_buffer)} tool calls from "
                                 f"stream")

                    for idx, tool_call_data in tool_calls_buffer.items():
                        func_name = tool_call_data["function"]["name"]
                        call_id = tool_call_data["id"]
                        args_str = tool_call_data["function"]["arguments"]

                        logger.debug(f"Stream tool call {idx}: {func_name}("
                                     f"{args_str[:100]}...)")

                        # Add tool call params to history
                        tool_params = ToolCallsParams(
                            call_id=call_id,
                            function=Function(name=func_name, arguments=args_str),
                        )
                        msg_list.append(tool_params)
                        logger.debug(f"Added stream tool call params: {call_id}")

                        # Execute tool and add result
                        if func_name in tools_map:
                            logger.debug(f"Executing stream tool: {func_name}")
                            args = json.loads(args_str)

                            start_time = time.time()
                            result = tools_map[func_name](**args)
                            exec_time = time.time() - start_time

                            logger.debug(f"Stream tool {func_name} executed in "
                                         f"{exec_time:.3f}s")

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
                            logger.debug(f"Added stream tool result: {call_id}")
                        else:
                            logger.debug(f"Stream tool {func_name} not found")
                            error_result = ToolCalls(call_id=call_id,
                                                     content=f"Tool {func_name} not available")
                            msg_list.append(error_result)

                    logger.debug(f"Continuing stream to next iteration with "
                                 f"{len(msg_list)} total messages")
                    continue  # Continue the loop for next iteration

                # No tool calls, we're done
                logger.debug(f"Stream completion finished after {iteration} iterations")
                break

        return AsyncStream(_stream_gen())

    @staticmethod
    def _build_msgs(
        messages: t.List[Message | ToolCalls | ToolCallsParams]
    ) -> t.List[chat_t.ChatCompletionMessageParam]:
        """Build OpenAI message list from Message objects."""
        logger.debug(f"Building OpenAI messages from {len(messages)} input messages")

        msgs = []  # type: t.List[chat_t.ChatCompletionMessageParam]
        msg_types = [type(msg).__name__ for msg in messages]
        logger.debug(f"Input message types: {msg_types}")

        for i, msg in enumerate(messages):
            if isinstance(msg, Message):
                if msg.role == "user":
                    msgs.append(
                        chat_t.ChatCompletionUserMessageParam(role="user", content=msg.content))
                    logger.debug(f"Built user message {i}: {len(msg.content)} chars")
                elif msg.role == "assistant":
                    msgs.append(
                        chat_t.ChatCompletionAssistantMessageParam(role="assistant",
                                                                   content=msg.content))
                    logger.debug(f"Built assistant message {i}: {len(msg.content)} chars")
            elif isinstance(msg, ToolCallsParams):
                logger.debug(f"Processing tool call params {i}: {msg.function.name}")

                # Look for existing assistant message to add tool calls to
                if msgs and msgs[-1].get("role") == "assistant":
                    # Add tool calls to the last assistant message
                    if "tool_calls" not in msgs[-1]:
                        msgs[-1]["tool_calls"] = []
                    msgs[-1]["tool_calls"].append(
                        chat_t.ChatCompletionMessageFunctionToolCallParam(
                            id=msg.call_id,
                            function=fn_param.Function(name=msg.function.name,
                                                       arguments=msg.function.arguments),
                            type="function",
                        ))
                    logger.debug(f"Added tool call to existing assistant message: "
                                 f"{msg.function.name}")
                else:
                    # Create new assistant message with tool calls
                    msgs.append(
                        chat_t.ChatCompletionAssistantMessageParam(
                            role="assistant",
                            tool_calls=[
                                chat_t.ChatCompletionMessageFunctionToolCallParam(
                                    id=msg.call_id,
                                    function=fn_param.Function(name=msg.function.name,
                                                               arguments=msg.function.arguments),
                                    type="function",
                                )
                            ],
                        ))
                    logger.debug(f"Created new assistant message with tool call: "
                                 f"{msg.function.name}")
            elif isinstance(msg, ToolCalls):
                msgs.append(
                    chat_t.ChatCompletionToolMessageParam(role="tool",
                                                          content=msg.content,
                                                          tool_call_id=msg.call_id))
                content_preview = (str(msg.content)[:100]
                                   if isinstance(msg.content, str) else str(msg.content))
                logger.debug(f"Built tool result message {i}: {content_preview}...")

        logger.debug(f"Built {len(msgs)} OpenAI messages")
        return msgs

    @staticmethod
    async def _parse_stream(completion: t.AsyncIterable[chat_t.ChatCompletionChunk],
                            chatcmpl_id: str, created: int,
                            model: str) -> AsyncStream[CompletionChunk]:
        logger.debug(f"Parsing stream with chatcmpl_id={chatcmpl_id}, model={model}")

        async def _gen() -> t.AsyncGenerator[CompletionChunk, None]:
            chunk_count = 0
            async for chunk in completion:
                chunk_count += 1
                choice = chunk.choices[0] if chunk.choices else None
                if not choice:
                    logger.debug(f"Stream chunk {chunk_count}: no choice data")
                    continue

                delta = choice.delta
                logger.debug(f"Stream chunk {chunk_count}: content="
                             f"{bool(delta.content)}, "
                             f"finish_reason={choice.finish_reason}")

                yield CompletionChunk(id=chatcmpl_id,
                                      delta=CompletionChunkDelta(content=delta.content,
                                                                 refusal=delta.refusal,
                                                                 reason=None),
                                      finish_reason=(map_finish_reason(choice.finish_reason)
                                                     if choice.finish_reason else None),
                                      created=created,
                                      model=model,
                                      usage=Usage(
                                          completion_tokens=chunk.usage.completion_tokens,
                                          prompt_tokens=chunk.usage.prompt_tokens,
                                          total_tokens=chunk.usage.total_tokens,
                                      ) if chunk.usage else None)
            logger.debug(f"Stream parsing completed after {chunk_count} chunks")

        return AsyncStream(_gen())
