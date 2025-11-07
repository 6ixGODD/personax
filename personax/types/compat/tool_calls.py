from __future__ import annotations

from personax.types import BaseSchema


class Function(BaseSchema):
    """Tool function call specification.

    Represents a function call request from the LLM during tool calling,
    containing the function name and serialized arguments.

    Attributes:
        name: The name of the function to call.
        arguments: JSON-serialized string of function arguments.

    Example:
        ```python
        func = Function(
            name="get_weather",
            arguments='{"location": "San Francisco", "unit": "celsius"}'
        )

        # Parse arguments for execution
        import json
        args = json.loads(func.arguments)
        result = await get_weather(**args)
        ```
    """

    name: str
    arguments: str

    __slots__ = ("arguments", "name")

    def __init__(
        self,
        *,
        name: str,
        arguments: str,
    ) -> None:
        self.name = name
        self.arguments = arguments


class ToolCallsParams(BaseSchema):
    """Tool call parameters from LLM response.

    Represents a tool call request emitted by the LLM, containing the
    call ID for tracking and the function specification. This is added
    to the conversation history before executing the tool.

    Attributes:
        call_id: Unique identifier for this tool call, used to match
            with the corresponding ToolCalls result.
        function: The function call specification.

    Example:
        ```python
        # LLM requests a tool call
        tool_params = ToolCallsParams(
            call_id="call_abc123",
            function=Function(
                name="get_weather",
                arguments='{"location": "Paris"}'
            )
        )

        # Add to message history
        messages.append(tool_params)

        # Execute tool
        args = json.loads(tool_params.function.arguments)
        result = await tools[tool_params.function.name](**args)

        # Add result to history
        messages.append(ToolCalls(
            call_id=tool_params.call_id,
            content=result
        ))
        ```
    """

    call_id: str
    function: Function

    __slots__ = ("call_id", "function")

    def __init__(
        self,
        *,
        call_id: str,
        function: Function,
    ) -> None:
        self.call_id = call_id
        self.function = function


class ToolCalls(BaseSchema):
    """Tool execution result.

    Represents the result of executing a tool call, containing the call ID
    for matching with the request and the execution result. This is added
    to the conversation history after tool execution.

    Attributes:
        call_id: Unique identifier matching the ToolCallsParams that
            requested this execution.
        content: The tool execution result, either as a string or list
            of strings for multiple results.

    Example:
        ```python
        # Single result
        result = ToolCalls(
            call_id="call_abc123",
            content="Current weather in Paris: 18°C, Partly cloudy"
        )

        # Multiple results (e.g., from batch operation)
        results = ToolCalls(
            call_id="call_def456",
            content=[
                '{"location": "Paris", "temp": 18}',
                '{"location": "London", "temp": 15}',
            ]
        )

        # Add to message history for next LLM call
        messages.append(result)
        ```
    """

    call_id: str
    content: str | list[str]

    __slots__ = ("call_id", "content")

    def __init__(
        self,
        *,
        call_id: str,
        content: str | list[str],
    ) -> None:
        self.call_id = call_id
        self.content = content
