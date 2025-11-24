from __future__ import annotations

import datetime
import logging
import typing as t
import zoneinfo

import typing_extensions as te

from personax.context import ContextSystem
from personax.exceptions import RESTError
from personax.resource.restful.ip import IpLocationService
from personax.resource.restful.ip import Location
from personax.resource.template import Template
from personax.types.context import Context

logger = logging.getLogger("personax.context.profile")


class ProfileContext(te.TypedDict):
    """User profile context with timestamp and location information.

    Contains user session metadata including preferred name, IP-based
    geolocation, timezone, and platform information. Used as the
    standard profile structure for PersonaX-based systems.
    """

    prefname: str | None
    """User's preferred display name."""

    ip: str | None
    """IP address of the user's current session."""

    location: Location | None
    """Geolocation data derived from IP address."""

    timestamp: str
    """Current timestamp in ISO format with timezone."""

    timezone: str
    """IANA timezone string (e.g., "America/New_York")."""

    user_agent: str | None
    """Browser/client user agent string."""

    platform: str | None
    """Operating system or platform identifier."""

    extras: te.NotRequired[t.Mapping[str, t.Any]]
    """Optional additional metadata fields."""


class Info(te.TypedDict, total=False):
    """User information input for profile context building.

    Defines the expected user data structure that should be provided in
    the context dictionary under ProfileContextSystem.__infokey__. All
    fields are optional.
    """

    prefname: str
    """User's preferred display name."""

    ip: str
    """IP address for geolocation lookup."""

    user_agent: str
    """Browser/client user agent string."""

    platform: str
    """Operating system or platform identifier."""

    timezone: str
    """IANA timezone string for timestamp generation."""


class ProfileContextSystem(ContextSystem[ProfileContext]):
    """Basic profile context system for user session information.

    Builds user profile context from provided user information, including
    timezone-aware timestamps and optional IP-based geolocation. Serves as the
    foundation for more specialized profile systems.

    The system requires user information to be provided in the input context
    under the __infokey__. Missing information results in None values in the
    profile context.

    Attributes:
        __key__: Context system identifier ("profile").
        __infokey__: Key for accessing user info in context ("profile.info").
        ip_service: Optional IP geolocation service.
        template: Template for rendering profile context to text.

    Args:
        ip_service: Optional service for IP-based geolocation lookup.
        template: Template for formatting profile context into text.

    Example:
        ```python
        from personax.resources.restful.ip.baidu import (
            BaiduIpLocationService,
        )

        profile_system = ProfileContextSystem(
            ip_service=BaiduIpLocationService(ak="YOUR_KEY"),
            template=Template(
                "User: {{ prefname or 'Guest' }}\n"
                "Time: {{ timestamp }}\n"
                "Location: {{ location.city if location else 'Unknown' }}"
            ),
        )

        # Provide user info in context
        context = Context(
            messages=[Message(role="user", content="Hello")],
            context={
                "profile.info": {
                    "prefname": "Alice",
                    "ip": "123.45.67.89",
                    "timezone": "America/New_York",
                }
            },
        )

        # Build profile context
        profile = await profile_system.build(context)
        # {
        #     "prefname": "Alice",
        #     "ip": "123.45.67.89",
        #     "location": {"city": "New York", ...},
        #     "timestamp": "2025-11-07T01:30:34-05:00",
        #     "timezone": "America/New_York",
        #     ...
        # }

        # Format to text
        text = await profile_system.parse(profile)
        # "User: Alice\nTime: 2025-11-07T01:30:34-05:00\nLocation: New York"
        ```

    Note:
        - User info must be provided under __infokey__ for proper functionality
        - IP geolocation is optional and gracefully degrades on failure
        - Timestamps are generated in the user's specified timezone
        - Falls back to UTC if timezone is invalid or not provided
        - Can be extended for domain-specific profile information
    """

    __key__ = "profile"
    __infokey__: t.ClassVar[str] = "profile.info"

    def __init__(
        self,
        *,
        ip_service: IpLocationService | None = None,
        template: Template,
    ) -> None:
        self.ip_service = ip_service
        self.template = template

    async def build(self, context: Context | str) -> ProfileContext:
        """Build profile context from user information.

        Extracts user data from the context dictionary, generates a
        timezone-aware timestamp, performs optional IP geolocation,
        and constructs the complete profile context.

        Args:
            context: Conversation context containing user info under __infokey__.

        Returns:
            ProfileContext with user session information and metadata.

        Example:
            ```python
            profile = await system.build(context)
            print(profile["prefname"])  # "Alice"
            print(profile["timestamp"])  # "2025-11-07T01:30:34-05:00"
            print(profile["location"]["city"])  # "New York"
            ```

        Note:
            - Uses user's timezone for timestamp, defaults to UTC on error
            - IP geolocation failures are silently ignored (location=None)
            - Missing user info results in None values in profile
        """
        # Get basic information
        info = context.context.get(self.__infokey__, Info())
        # Get current timestamp in the specified timezone
        try:
            tz = zoneinfo.ZoneInfo(info.get("timezone", "UTC") or "UTC")
        except Exception:
            tz = zoneinfo.ZoneInfo("UTC")
        timestamp = datetime.datetime.now(tz).isoformat()

        # Get location from IP if available
        location = None  # type: Location | None
        if info.get("ip") and self.ip_service:
            try:
                location = await self.ip_service.locate(info.get("ip"))
            except RESTError:
                # If IP location service fails, continue without location
                location = None

        logger.debug(
            "Built ProfileContext: prefname=%s, ip=%s, location=%s, timestamp=%s, timezone=%s, user_agent=%s, platform=%s",
            info.get("prefname"),
            info.get("ip"),
            location,
            timestamp,
            info.get("timezone", "UTC") or "UTC",
            info.get("user_agent"),
            info.get("platform"),
        )

        return ProfileContext(
            prefname=info.get("prefname"),
            ip=info.get("ip"),
            location=location,
            timestamp=timestamp,
            timezone=info.get("timezone", "UTC") or "UTC",
            user_agent=info.get("user_agent"),
            platform=info.get("platform"),
        )

    async def parse(self, built: ProfileContext) -> str:
        """Format profile context into text using the configured
        template.

        Args:
            built: Profile context from build().

        Returns:
            Formatted text representation of the profile.
        """
        return self.template.render(context=built)
