from __future__ import annotations

import abc

import typing_extensions as te

from personax.resources.rest import RESTResource


class Location(te.TypedDict):
    address: str
    """Human-readable address."""

    adcode: str
    """Administrative division code."""


class IpLocationService(RESTResource):
    @abc.abstractmethod
    async def locate(self, ip: str, /) -> Location:
        pass
