from __future__ import annotations

import typing as t

from personax.types import BaseSchema


class Function(BaseSchema):
    name: str
    arguments: str

    __slots__ = ("name", "arguments")

    def __init__(
        self,
        *,
        name: str,
        arguments: str,
    ) -> None:
        self.name = name
        self.arguments = arguments


class ToolCallsParams(BaseSchema):
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
    call_id: str
    content: str | t.List[str]

    __slots__ = ("call_id", "content")

    def __init__(
        self,
        *,
        call_id: str,
        content: str | t.List[str],
    ) -> None:
        self.call_id = call_id
        self.content = content
