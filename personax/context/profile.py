from __future__ import annotations

import datetime
import typing as t
import zoneinfo

import typing_extensions as te

from personax.context import ContextSystem
from personax.exceptions import RESTResourceException
from personax.resources.rest.ip import IpLocationService, Location
from personax.resources.template import Template
from personax.types.context import Context


class ProfileContext(te.TypedDict):
    prefname: str | None
    """Current user's preferred name."""

    ip: str | None
    """Current user's IP address."""

    location: Location | None
    """IP geolocation information."""

    timestamp: str
    """Current timestamp in ISO format."""

    timezone: str
    """Current timezone. Should be a valid IANA timezone string."""

    user_agent: str | None
    """User agent string."""

    platform: str | None
    """Platform/OS information."""

    extras: te.NotRequired[t.Mapping[str, t.Any]]
    """Extra information."""


class Info(te.TypedDict, total=False):
    prefname: str
    """Current user's preferred name."""

    ip: str
    """Current user's IP address."""

    user_agent: str
    """User agent string."""

    platform: str
    """Platform/OS information."""

    timezone: str
    """Timezone string in IANA format, e.g., 'America/New_York'."""


class ProfileContextSystem(ContextSystem[ProfileContext]):
    __key__ = "profile"

    def __init__(
        self,
        *,
        ip_service: IpLocationService | None = None,
        template: Template,
        provide_info: t.Callable[[], Info],
    ) -> None:
        self.ip_service = ip_service
        self.template = template
        self.provide_info = provide_info

    async def build(self, context: Context | str) -> ProfileContext:
        # Get basic information
        info = self.provide_info()
        # Get current timestamp in the specified timezone
        try:
            tz = zoneinfo.ZoneInfo(info.get('timezone', 'UTC') or "UTC")
        except Exception:
            tz = zoneinfo.ZoneInfo("UTC")
        timestamp = datetime.datetime.now(tz).isoformat()

        # Get location from IP if available
        location = None  # type: Location | None
        if info.get('ip') and self.ip_service:
            try:
                location = await self.ip_service.locate(info.get('ip'))
            except RESTResourceException:
                # If IP location service fails, continue without location
                location = None

        return ProfileContext(prefname=info.get('prefname'),
                              ip=info.get('ip'),
                              location=location,
                              timestamp=timestamp,
                              timezone=info.get('timezone', 'UTC') or "UTC",
                              user_agent=info.get('user_agent'),
                              platform=info.get('platform'))

    async def parse(self, built: ProfileContext) -> str:
        return self.template.render(context=built)
