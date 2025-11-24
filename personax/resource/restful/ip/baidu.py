from __future__ import annotations

import logging
import typing as t

import async_lru as alru
import httpx
import pydantic as pydt
import typing_extensions as te

from personax.exceptions import RESTError
from personax.resource.restful.ip import IpLocationService
from personax.resource.restful.ip import Location

logger = logging.getLogger("personax.resources.rest.ip.baidu")


class BaiduLocationParams(te.TypedDict):
    """Request parameters for Baidu IP Location API."""

    ip: str
    """IP address to locate"""

    ak: str
    """Baidu API Access Key"""

    coor: str
    """Coordinate type, e.g., 'gcj02' for GCJ-02 coordinates (Mars
    coordinates)"""


class BaiduLocationAddressDetail(te.TypedDict):
    """Detailed address components from Baidu location response."""

    adcode: str
    """Administrative division code"""

    city: str
    """City name"""

    city_code: int
    """City code"""

    district: str
    """District name"""

    province: str
    """Province name"""

    street: str
    """Street name"""

    street_number: str
    """Street number"""


class BaiduLocationContent(te.TypedDict):
    """Location content from Baidu API response."""

    address: str
    """Formatted address"""

    address_detail: BaiduLocationAddressDetail
    """Detailed address components"""

    point: dict[str, t.Any]
    """Geographical coordinates (latitude and longitude)"""


class BaiduLocation(pydt.BaseModel):
    """Baidu IP Location API response model."""

    status: int
    """Response status code, 0 indicates success"""

    message: str | None = None
    """Response message, present when status is not 0"""

    address: str
    """Queried IP address"""

    content: BaiduLocationContent
    """Location content including address and coordinates"""


class BaiduIpLocationService(IpLocationService):
    """Baidu Map IP geolocation service implementation.

    Provides IP-to-location lookup using Baidu Maps API with automatic caching
    of results to reduce API calls and improve performance.

    The service uses GCJ-02 coordinate system (Chinese Mars coordinates) by
    default, which is the standard for most Chinese mapping services.

    Attributes:
        ak: Baidu API Access Key.
        max_retries: Maximum retry attempts for failed requests.
        retry_wait: Wait time between retries in seconds.

    Args:
        ak: Baidu API Access Key for authentication.
        timeout: Request timeout in seconds. Defaults to 10.0.
        max_retries: Maximum retry attempts. Defaults to 3.
        retry_wait: Wait time between retries. Defaults to 2.0.
        http_client: Optional pre-configured httpx client.

    Example:
        ```python
        service = BaiduIpLocationService(ak="YOUR_BAIDU_AK")

        location = await service.locate("123.45.67.89")
        print(location["address"])  # "北京市海淀区..."
        print(location["adcode"])  # "110108"

        # Results are cached - second call uses cache
        location2 = await service.locate("123.45.67.89")  # Instant
        ```

    Note:
        - Results are cached with LRU eviction (max 1024 entries)
        - Requires valid Baidu API Access Key
        - Returns Chinese addresses for Chinese IPs
        - Uses GCJ-02 coordinate system
    """
    def __init__(
        self,
        ak: str,
        *,
        timeout: float = 10.0,
        max_retries: int = 3,
        retry_wait: float = 2.0,
        http_client: httpx.AsyncClient | None = None,
    ):
        self.ak = ak
        super().__init__(
            base_url="https://api.map.baidu.com/location/", timeout=timeout, http_client=http_client
        )
        self.max_retries = max_retries
        self.retry_wait = retry_wait

    @alru.alru_cache(maxsize=1024)
    async def locate(self, ip: str, /) -> Location:
        """Lookup location from IP address with caching.

        Args:
            ip: IP address to locate.

        Returns:
            Location with address and administrative code.

        Raises:
            RESTError: If Baidu API returns an error status.

        Example:
            ```python
            location = await service.locate("114.114.114.114")
            print(location["address"])  # "江苏省南京市..."
            ```
        """
        params = BaiduLocationParams(ip=ip, ak=self.ak, coor="gcj02")
        response = await self.request(
            "ip",
            method="GET",
            params=params,
            max_retries=self.max_retries,
            cast_to=BaiduLocation,
            retry_wait=self.retry_wait,
        )
        if response.status != 0:
            logger.error(f"Baidu Location IP Service error for IP {ip}: {response.message}")
            raise RESTError(f"Baidu Location IP Service error: {response.message}")
        logger.debug(f"Baidu Location IP Service response for IP {ip}: {response}")
        return Location(
            address=response.content["address"], adcode=response.content["address_detail"]["adcode"]
        )
