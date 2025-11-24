from __future__ import annotations

import typing as t


class Function(t.NamedTuple):
    """Tool function call specification.

    Represents a function call request from the LLM during tool calling,
    containing the function name and serialized arguments.

    Attributes:
        name: The name of the function to call.
        arguments: JSON-serialized string of function arguments.
    """

    name: str
    arguments: str


class ToolCallsParams(t.NamedTuple):
    """Tool call parameters from LLM response.

    Represents a tool call request emitted by the LLM, containing the
    call ID for tracking and the function specification. This is added
    to the conversation history before executing the tool.

    Attributes:
        call_id: Unique identifier for this tool call, used to match
            with the corresponding ToolCalls result.
        function: The function call specification.
    """

    call_id: str
    function: Function


class ToolCalls(t.NamedTuple):
    """Tool execution result.

    Represents the result of executing a tool call, containing the call ID
    for matching with the request and the execution result. This is added
    to the conversation history after tool execution.

    Attributes:
        call_id: Unique identifier matching the ToolCallsParams that
            requested this execution.
        content: The tool execution result, either as a string or list
            of strings for multiple results.
    """

    call_id: str
    content: str | list[str]
