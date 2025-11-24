from __future__ import annotations

import abc
import functools as ft
import inspect
import json
import re
import typing as t

import pydantic as pydt

# JSON serializable types
JsonPrimitive = str | int | float | bool | None
JsonValue = JsonPrimitive | t.Mapping[str, t.Any] | t.Sequence[t.Any]
JsonObject = JsonValue | t.Mapping[str, JsonValue]
JsonArray = t.Sequence[JsonValue]

# JSON Schema types
JsonSchemaType = t.Literal["string", "integer", "number", "boolean", "array", "object"]


# ruff: noqa: N815
class PropertySchema(pydt.BaseModel):
    """JSON Schema property definition for tool parameters.

    Represents the schema for a single parameter in a tool function,
    following JSON Schema specification. Used to validate and document
    parameter types and constraints.

    Attributes:
        type: JSON Schema type of the property.
        description: Human-readable description of the parameter.
        enum: List of allowed values for enum types.
        items: Schema for array item types.
        properties: Nested properties for object types.
        required: Required property names for object types.
        minimum: Minimum value for numeric types.
        maximum: Maximum value for numeric types.
        minLength: Minimum string length.
        maxLength: Maximum string length.
        pattern: Regular expression pattern for string validation.
        format: String format hint (e.g., "date", "email").
        default: Default value if parameter is not provided.
        examples: Example values for documentation.
        minItems: Minimum array length.
        maxItems: Maximum array length.
        uniqueItems: Whether array items must be unique.
        additionalProperties: Schema for additional object properties.
    """

    type: JsonSchemaType
    description: str | None = None
    enum: list[JsonValue] | None = None
    items: dict[str, JsonValue] | None = None
    properties: dict[str, dict[str, JsonValue]] | None = None
    required: list[str] | None = None
    minimum: int | float | None = None
    maximum: int | float | None = None
    minLength: int | None = None
    maxLength: int | None = None
    pattern: str | None = None
    format: str | None = None
    default: JsonValue | None = None
    examples: list[JsonValue] | None = None
    minItems: int | None = None
    maxItems: int | None = None
    uniqueItems: bool | None = None
    additionalProperties: bool | dict[str, JsonValue] | None = None

    model_config = pydt.ConfigDict(extra="allow")


class ParameterSchema(pydt.BaseModel):
    """Schema for tool function parameters.

    Defines the complete parameter structure for a tool function,
    including all parameters and their constraints.
    """

    type: t.Literal["object"] = "object"
    """Always "object" for function parameters."""

    properties: dict[str, PropertySchema]
    """Dictionary mapping parameter names to their schemas."""

    required: list[str] = pydt.Field(default_factory=list)
    """List of required parameter names."""


class FunctionSchema(pydt.BaseModel):
    """Schema for a tool function definition.

    Represents the complete specification of a function that can be
    called by an LLM, including its name, description, and parameters.
    """

    name: t.Annotated[
        str,
        pydt.Field(
            min_length=1,
            max_length=64,
            description="The name of the function to be called by the tool.",
            examples=["get_weather", "calculate_sum"],
        ),
    ]
    """Function name (1-64 characters, used in tool calls)."""

    description: str
    """Description of what the function does."""

    parameters: ParameterSchema | None = None
    """Schema defining the function's input parameters."""


class ToolSchema(pydt.BaseModel):
    """Complete tool schema for LLM function calling.

    Top-level schema representing a tool that can be invoked by the LLM.
    Follows the OpenAI function calling specification.
    """

    type: t.Literal["function"] = "function"
    """Always "function" for function-based tools."""

    function: FunctionSchema
    """The function specification."""


# pylint: disable=too-few-public-methods
class Property:
    """Parameter property descriptor for tool function arguments.

    Used with typing.Annotated to attach metadata to function parameters.
    This metadata is extracted during schema generation to create rich
    parameter documentation and validation constraints.

    The Property class allows declarative definition of parameter constraints
    that are automatically converted to JSON Schema format when the tool
    schema is generated.

    Attributes:
        description: Human-readable parameter description.
        enums: List of allowed values (for enum-like parameters).
        minimum: Minimum value for numeric parameters.
        maximum: Maximum value for numeric parameters.
        min_length: Minimum string length.
        max_length: Maximum string length.
        pattern: Regular expression pattern for string validation.
        format: String format hint (e.g., "email", "uri").
        default: Default value if not provided.
        examples: Example values for documentation.
        min_items: Minimum array length.
        max_items: Maximum array length.
        unique_items: Whether array items must be unique.
        extra: Additional custom properties as keyword arguments.

    Example:
        ```python
        class GetWeather(BaseTool):
            async def __call__(
                self,
                location: t.Annotated[
                    str,
                    Property(
                        description="City name or coordinates",
                        examples=[
                            "San Francisco",
                            "40.7128,-74.0060",
                        ],
                        min_length=1,
                        max_length=100,
                    ),
                ],
                unit: t.Annotated[
                    t.Literal["celsius", "fahrenheit"],
                    Property(
                        description="Temperature unit",
                        default="celsius",
                    ),
                ] = "celsius",
            ) -> Weather: ...
        ```
    """

    __slots__ = (
        "default",
        "description",
        "enums",
        "examples",
        "extra",
        "format",
        "max_items",
        "max_length",
        "maximum",
        "min_items",
        "min_length",
        "minimum",
        "pattern",
        "unique_items",
    )

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        *,
        description: str | None = None,
        enums: list[JsonValue] | None = None,
        minimum: int | float | None = None,
        maximum: int | float | None = None,
        min_length: int | None = None,
        max_length: int | None = None,
        pattern: str | None = None,
        format: str | None = None,  # pylint: disable=redefined-builtin
        default: JsonValue | None = None,
        examples: list[JsonValue] | None = None,
        min_items: int | None = None,
        max_items: int | None = None,
        unique_items: bool | None = None,
        **kwargs: JsonValue,
    ):
        self.description = description
        self.enums = enums
        self.minimum = minimum
        self.maximum = maximum
        self.min_length = min_length
        self.max_length = max_length
        self.pattern = pattern
        self.format = format
        self.default = default
        self.examples = examples
        self.min_items = min_items
        self.max_items = max_items
        self.unique_items = unique_items
        self.extra = kwargs


def _get_json_schema_type(python_type: type) -> JsonSchemaType:
    """Map Python type to JSON Schema type.

    Args:
        python_type: Python type annotation.

    Returns:
        Corresponding JSON Schema type string.
    """
    origin = t.get_origin(python_type)
    args = t.get_args(python_type)

    result: JsonSchemaType

    if origin is t.Union:
        non_none_types = [arg for arg in args if arg is not type(None)]
        target = non_none_types[0] if non_none_types else str
        result = _get_json_schema_type(target)

    elif origin is list or python_type is list:
        result = "array"

    elif origin is dict or python_type is dict:
        result = "object"

    elif origin is t.Literal:
        first_arg = args[0]
        if isinstance(first_arg, str):
            result = "string"
        elif isinstance(first_arg, int):
            result = "integer"
        elif isinstance(first_arg, float):
            result = "number"
        elif isinstance(first_arg, bool):
            result = "boolean"
        else:
            result = "string"

    else:
        type_mapping: dict[type, JsonSchemaType] = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            dict: "object",
            list: "array",
        }
        result = type_mapping.get(python_type, "string")

    return result


def _get_array_items_schema(python_type: type) -> dict[str, JsonValue] | None:
    """Extract array item schema from list type annotation.

    Args:
        python_type: List type annotation (e.g., list[str]).

    Returns:
        JSON Schema for array items, or None if not an array type.
    """
    origin = t.get_origin(python_type)
    args = t.get_args(python_type)

    if origin is list and args:
        item_type = args[0]
        return {"type": _get_json_schema_type(item_type)}

    return None


def _get_literal_enum_values(python_type: type) -> list[JsonValue] | None:
    """Extract enum values from Literal type annotation.

    Args:
        python_type: Literal type annotation (e.g., Literal["a", "b"]).

    Returns:
        List of literal values, or None if not a Literal type.
    """
    origin = t.get_origin(python_type)
    args = t.get_args(python_type)

    if origin is t.Literal:
        return list(args)

    return None


def _extract_property_from_annotation(
    annotation: t.Any,
) -> tuple[type, Property | None]:
    """Extract type and Property descriptor from Annotated type.

    Args:
        annotation: Type annotation, possibly Annotated with Property.

    Returns:
        Tuple of (actual_type, property_descriptor). Property is None if not
        annotated with Property.
    """
    if t.get_origin(annotation) is t.Annotated:
        args = t.get_args(annotation)
        actual_type = args[0]

        for metadata in args[1:]:
            if isinstance(metadata, Property):
                return actual_type, metadata

        return actual_type, None

    return annotation, None


_ReturnType: t.TypeAlias = str | JsonObject | t.Sequence[JsonObject] | None
ReturnType: t.TypeAlias = _ReturnType | t.Awaitable[_ReturnType]
P = t.ParamSpec("P")
R = t.TypeVar("R", bound=ReturnType)


class BaseTool(t.Generic[P, R], abc.ABC):
    """Abstract base class for LLM-callable tools.

    BaseTool provides a declarative framework for defining functions that can
    be invoked by language models. Tools are defined as classes with typed
    __call__ methods, and their schemas are automatically generated from type
    annotations and Property descriptors.

    Key Features:
    - Automatic schema generation from type hints
    - Rich parameter documentation via Property annotations
    - Support for both sync and async implementations
    - Automatic snake_case naming from class names
    - Full JSON Schema compliance for LLM integration

    Type Parameters:
        P: Parameter specification for __call__ method.
        R: Return type (str, dict, list of dicts, or None, optionally async).

    Class Variables:
        __function_description__: Required description of what the tool does.

    Generated Properties:
        __function_name__: Auto-generated snake_case name from class name.
        schema: ToolSchema object with complete function specification.
        schema_dict: Schema as dictionary for API submission.
        schema_json: Schema as JSON string for inspection.

    Example:
        ```python
        # Define a tool
        class CalculateSum(BaseTool):
            __function_description__ = "Add two numbers together"

            async def __call__(
                self,
                a: t.Annotated[
                    int,
                    Property(
                        description="First number",
                        minimum=0,
                        maximum=1000,
                    ),
                ],
                b: t.Annotated[
                    int,
                    Property(
                        description="Second number",
                        minimum=0,
                        maximum=1000,
                    ),
                ],
            ) -> str:
                return str(a + b)


        # Use the tool
        tool = CalculateSum()
        print(tool.__function_name__)  # "calculate_sum"
        print(tool.schema_json)  # Full JSON Schema

        # Execute
        result = await tool(a=5, b=3)  # "8"


        # With CompletionSystem
        completion = await system.complete(
            messages,
            model="gpt-4",
            tools=[CalculateSum()],
        )


        # Complex example with multiple parameter types
        class SearchDatabase(BaseTool):
            __function_description__ = (
                "Search database with filters"
            )

            def __init__(self, db_client):
                self.db = db_client

            async def __call__(
                self,
                query: t.Annotated[
                    str,
                    Property(
                        description="Search query string",
                        min_length=1,
                        max_length=200,
                        examples=["user:john", "status:active"],
                    ),
                ],
                limit: t.Annotated[
                    int,
                    Property(
                        description="Maximum results to return",
                        minimum=1,
                        maximum=100,
                        default=10,
                    ),
                ] = 10,
                fields: t.Annotated[
                    list[str] | None,
                    Property(
                        description="Fields to return",
                        examples=[["name", "email"]],
                    ),
                ] = None,
            ) -> list[dict[str, t.Any]]:
                return await self.db.search(query, limit, fields)
        ```

    Note:
        - __call__ must be defined by subclasses
        - __function_description__ must be set as a class variable
        - Parameter types should use Property for rich documentation
        - Return type can be sync or async
        - Tools are hashable by function name for use in sets/dicts
    """

    __function_description__: t.ClassVar[str]

    @abc.abstractmethod
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """Execute the tool with given arguments.

        Must be implemented by subclasses. Can be sync or async.

        Args:
            *args: Positional arguments matching parameter schema.
            **kwargs: Keyword arguments matching parameter schema.

        Returns:
            Tool result as string, dict, list of dicts, or None.
            Can be wrapped in Awaitable for async tools.
        """

    @staticmethod
    def _parse_arguments(args: str) -> dict[str, JsonValue]:
        """Parse JSON string arguments into dictionary.

        Args:
            args: JSON-serialized argument string from LLM.

        Returns:
            Dictionary of parsed arguments.

        Raises:
            ValueError: If JSON parsing fails.
        """
        if not args:
            return {}
        try:
            return t.cast(dict[str, JsonValue], json.loads(args))
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON string: {args}") from e

    @ft.cached_property
    def __function_name__(self) -> str:
        """Generate snake_case function name from class name.

        Converts class name like "GetWeather" to "get_weather".

        Returns:
            Snake case function name.

        Example:
            ```python
            class GetUserProfile(BaseTool): ...


            tool = GetUserProfile()
            print(tool.__function_name__)  # "get_user_profile"
            ```
        """
        cls_name = self.__class__.__name__
        s1 = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", cls_name)
        s2 = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", s1)
        return s2.lower().replace(" ", "_").replace("-", "_")

    @ft.cached_property
    def schema(self) -> ToolSchema:
        """Generate complete tool schema from __call__ signature.

        Inspects the __call__ method's type hints and Property annotations
        to build a comprehensive JSON Schema-compliant tool specification.

        Returns:
            ToolSchema with complete function specification.

        Example:
            ```python
            tool = GetWeather()
            schema = tool.schema

            print(schema.function.name)  # "get_weather"
            print(
                schema.function.description
            )  # "Get weather for location"
            print(schema.function.parameters.required)  # ["location"]
            ```
        """
        sig = inspect.signature(self.__call__)
        properties: dict[str, PropertySchema] = {}
        required: list[str] = []
        type_hints = t.get_type_hints(self.__call__, include_extras=True)

        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue

            actual_annotation = type_hints[param_name]
            # Extract type and Property descriptor
            actual_type, prop_desc = _extract_property_from_annotation(actual_annotation)

            # Get basic JSON schema type
            json_type = _get_json_schema_type(actual_type)

            # Build PropertySchema directly
            property_schema = PropertySchema(
                type=json_type,
                description=prop_desc.description if prop_desc else None,
                enum=(
                    prop_desc.enums
                    if prop_desc and prop_desc.enums
                    else _get_literal_enum_values(actual_type)
                ),
                items=_get_array_items_schema(actual_type),
                minimum=prop_desc.minimum if prop_desc else None,
                maximum=prop_desc.maximum if prop_desc else None,
                minLength=prop_desc.min_length if prop_desc else None,
                maxLength=prop_desc.max_length if prop_desc else None,
                pattern=prop_desc.pattern if prop_desc else None,
                format=prop_desc.format if prop_desc else None,
                default=prop_desc.default if prop_desc else None,
                examples=prop_desc.examples if prop_desc else None,
                minItems=prop_desc.min_items if prop_desc else None,
                maxItems=prop_desc.max_items if prop_desc else None,
                uniqueItems=prop_desc.unique_items if prop_desc else None,
            )

            # Add any extra properties from Property descriptor
            if prop_desc and prop_desc.extra:
                for key, value in prop_desc.extra.items():
                    setattr(property_schema, key, value)

            properties[param_name] = property_schema

            # Check if parameter is required (no default value)
            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        return ToolSchema(
            function=FunctionSchema(
                name=self.__function_name__,
                description=self.__function_description__,
                parameters=(
                    ParameterSchema(properties=properties, required=required)
                    if properties
                    else None
                ),
            )
        )

    @property
    def schema_dict(self) -> dict[str, JsonValue]:
        """Convert tool schema to dictionary format.

        Returns:
            Schema as dictionary, suitable for API submission.

        Example:
            ```python
            tool = GetWeather()
            schema_dict = tool.schema_dict

            # Pass to OpenAI
            completion = client.chat.completions.create(
                messages=[...],
                tools=[schema_dict],
            )
            ```
        """
        return self.schema.model_dump(exclude_none=True, mode="json")

    @property
    def schema_json(self) -> str:
        """Convert tool schema to JSON string.

        Returns:
            Pretty-printed JSON schema string.

        Example:
            ```python
            tool = GetWeather()
            print(tool.schema_json)
            # {
            #   "type": "function",
            #   "function": {
            #     "name": "get_weather",
            #     ...
            #   }
            # }
            ```
        """
        return self.schema.model_dump_json(indent=2, exclude_none=True)

    def __hash__(self) -> int:
        """Hash tool by function name for use in sets/dicts.

        Returns:
            Hash of the function name.
        """
        return hash(self.__function_name__)


BaseToolType: t.TypeAlias = BaseTool[t.Any, ReturnType]
