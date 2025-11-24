from __future__ import annotations

import os
import typing as t

import pydantic as pyd


class BaseModel(pyd.BaseModel):
    """Base model with common configuration for all Pydantic models.

    This class sets default configurations such as validation on assignment,
    forbidding extra fields, allowing arbitrary types, and making the model
    immutable.

    Attributes:
        model_config: Configuration dictionary for the model.
    """
    model_config: t.ClassVar[pyd.ConfigDict] = pyd.ConfigDict(
        validate_assignment=True,  # Validate on assignment
        validate_default=False,  # Do not validate default values
        extra="forbid",  # Disallow extra fields
        arbitrary_types_allowed=True,  # Allow arbitrary types
        populate_by_name=True,  # Allow population by field name
        use_enum_values=True,  # Use enum values directly
        frozen=True,  # Make the model immutable
        serialize_by_alias=True,  # Serialize using field aliases
    )


PathLikes: t.TypeAlias = str | os.PathLike[str]
"""Type alias for path-like objects."""
