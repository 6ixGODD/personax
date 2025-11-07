from __future__ import annotations


class PersonaXError(Exception):
    """Base exception for PersonaX-related errors."""

    def __init__(self, msg: str, /):
        super().__init__(msg)
        self.message = msg

    def __str__(self) -> str:
        return self.message

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.message!r})"


class ToolCallError(PersonaXError):
    """Exception raised for errors during tool calls."""


class ResourceError(PersonaXError):
    """Exception raised for resource-related errors."""


class RESTError(ResourceError):
    """Exception raised for REST resource-related errors."""
