from __future__ import annotations


class PersonaXException(Exception):
    """Base exception for PersonaX-related errors."""

    def __init__(self, msg: str, /):
        super().__init__(msg)
        self.message = msg

    def __str__(self) -> str:
        return self.message

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.message!r})"


class ToolCallException(PersonaXException):
    """Exception raised for errors during tool calls."""
    pass


class ResourceException(PersonaXException):
    """Exception raised for resource-related errors."""
    pass


class RESTResourceException(ResourceException):
    """Exception raised for REST resource-related errors."""
    pass
