from __future__ import annotations

import abc

import typing_extensions as te

from personax.resource.restful import RESTfulMixin


class Location(te.TypedDict):
    """Geographic location data structure.

    Contains location information derived from IP address or other
    sources.
    """

    address: str
    """Human-readable address string."""

    adcode: str
    """Administrative division code for the location."""


class IpLocationService(RESTfulMixin, abc.ABC):
    """Abstract base class for IP geolocation services.

    Defines the interface for IP-to-location lookup services. Implementations
    should integrate with specific geolocation APIs (e.g., Baidu, ipapi).

    Example:
        ```python
        class CustomIpService(IpLocationService):
            async def locate(self, ip: str) -> Location:
                response = await self.request(
                    f"/lookup/{ip}",
                    method="GET",
                    cast_to=CustomLocationResponse,
                )
                return Location(
                    address=response.full_address,
                    adcode=response.admin_code,
                )
        ```
    """

    @abc.abstractmethod
    async def locate(self, ip: str, /) -> Location:
        """Lookup geographic location from IP address.

        Args:
            ip: IP address to locate (IPv4 or IPv6).

        Returns:
            Location data including address and administrative code.

        Raises:
            RESTError: If the geolocation service request fails.
        """
