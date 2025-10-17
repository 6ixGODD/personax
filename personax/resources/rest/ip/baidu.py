from __future__ import annotations

import typing as t

import httpx
import pydantic as pydt
import typing_extensions as te
import async_lru as alru

from personax.exceptions import RESTResourceException
from personax.resources.rest.ip import IpLocationService
from personax.resources.rest.ip import Location


class BaiduLocationParams(te.TypedDict):
    ip: str
    """IP address to locate"""

    ak: str
    """Baidu API Access Key"""

    coor: str
    """Coordinate type, e.g., 'gcj02' for GCJ-02 coordinates (Mars 
    coordinates)"""


class BaiduLocationAddressDetail(te.TypedDict):
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
    address: str
    """Formatted address"""

    address_detail: BaiduLocationAddressDetail
    """Detailed address components"""

    point: dict[str, t.Any]
    """Geographical coordinates (latitude and longitude)"""


class BaiduLocation(pydt.BaseModel):
    status: int
    """Response status code, 0 indicates success"""

    message: str | None = None
    """Response message, present when status is not 0"""

    address: str
    """Queried IP address"""

    content: BaiduLocationContent
    """Location content including address and coordinates"""


class BaiduIpLocationService(IpLocationService):

    def __init__(
        self,
        ak: str,
        *,
        timeout: float = 10.0,
        max_retries: int = 3,
        retry_wait: float = 2.0,
        http_client: httpx.AsyncClient | None = None
    ):
        self.ak = ak
        super().__init__(
            base_url="https://api.map.baidu.com/location/",
            timeout=timeout,
            http_client=http_client
        )
        self.max_retries = max_retries
        self.retry_wait = retry_wait

    @alru.alru_cache(maxsize=1024)
    async def locate(self, ip: str, /) -> Location:
        params = BaiduLocationParams(ip=ip, ak=self.ak, coor="gcj02")
        response = await self.request(
            "ip",
            method="GET",
            params=params,
            max_retries=self.max_retries,
            cast_to=BaiduLocation,
            retry_wait=self.retry_wait
        )
        if response.status != 0:
            raise RESTResourceException(f"Baidu Location IP Service error: {response.message}")
        return Location(
            address=response.content["address"],
            adcode=response.content["address_detail"]["adcode"]
        )
