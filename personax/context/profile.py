from __future__ import annotations

import dataclasses as dc
import datetime
import typing as t

import typing_extensions as te

from personax.context import ContextSystem
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
    """Current timezone."""

    user_agent: str | None
    """User agent string."""

    platform: str | None
    """Platform/OS information."""

    extras: te.NotRequired[t.Mapping[str, t.Any]]
    """Extra information."""


@dc.dataclass
class Info:
    prefname: str | None = dc.field(default=None)
    ip: str | None = dc.field(default=None)
    user_agent: str | None = dc.field(default=None)
    platform: str | None = dc.field(default=None)
    timezone: str = dc.field(default='UTC')


class ProfileContextSystem(ContextSystem[ProfileContext]):

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
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # Get location from IP if available
        location = None  # type: Location | None
        if info.ip and self.ip_service:
            try:
                location = await self.ip_service.locate(info.ip)
            except Exception:
                # If IP location service fails, continue without location
                location = None

        return ProfileContext(prefname=info.prefname,
                              ip=info.ip,
                              location=location,
                              timestamp=timestamp,
                              timezone=info.timezone,
                              user_agent=info.user_agent,
                              platform=info.platform)

    async def parse(self, built: ProfileContext) -> str:
        return self.template.render(context=built)

    async def init(self) -> None:
        pass

    async def close(self) -> None:
        pass
