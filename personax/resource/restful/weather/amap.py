from __future__ import annotations

import logging
import typing as t

import pydantic as pydt
import typing_extensions as te

from personax.exceptions import RESTError
from personax.resource.restful.weather import WeatherInfo
from personax.resource.restful.weather import WeatherInfoService

logger = logging.getLogger("personax.resources.rest.weather.amap")


class AmapWeatherInfoParams(te.TypedDict):
    """Request parameters for Amap Weather API."""

    key: str
    """API key for Amap Weather Service"""

    city: str
    """Adcode of the city, e.g., '110000' for Beijing"""

    extensions: t.Literal["base"]
    """Type of weather data: 'base' for current weather"""

    output: t.Literal["JSON"]
    """Response format: 'JSON'"""


class AmapWeatherInfoLives(te.TypedDict):
    """Live weather data from Amap API response."""

    province: str
    """Province name"""

    city: str
    """City name"""

    adcode: str
    """Adcode of the city"""

    weather: str
    """Weather description"""

    temperature: str
    """Temperature in Celsius"""

    winddirection: str
    """Wind direction in degrees"""

    windpower: str
    """Wind power level"""

    humidity: str
    """Humidity percentage"""

    reporttime: str
    """Report time in ISO 8601 format"""


class AmapWeatherInfo(pydt.BaseModel):
    """Amap Weather API response model."""

    status: str = pydt.Field(..., description='Response status, "1" indicates success')

    count: str | None = pydt.Field(None, description="Number of results returned")

    info: str | None = pydt.Field(None, description='Response information, "OK" indicates success')

    infocode: str | None = pydt.Field(None, description='Response code, "10000" indicates success')

    lives: list[AmapWeatherInfoLives] = pydt.Field(
        default_factory=list,
        description="List of live weather information",
    )


class AmapWeatherInfoService(WeatherInfoService):
    """Amap weather information service implementation.

    Provides real-time weather data for Chinese locations using Amap (Gaode
    Maps) Weather API. Requires a valid Amap API key.

    Attributes:
        key: Amap API key for authentication.
        max_retries: Maximum retry attempts for failed requests.
        retry_wait: Wait time between retries in seconds.

    Args:
        key: Amap API key.
        timeout: Request timeout in seconds. Defaults to 5.0.
        max_retries: Maximum retry attempts. Defaults to 3.
        retry_wait: Wait time between retries. Defaults to 2.0.

    Example:
        ```python
        service = AmapWeatherInfoService(key="YOUR_AMAP_KEY")

        weather = await service.fetch("110000")  # Beijing
        print(weather["address"])  # "北京市 北京市"
        print(weather["temperature"])  # "15"
        print(weather["condition"])  # "晴"
        print(weather["humidity"])  # "45"
        ```

    Note:
        - Requires valid Amap API key
        - Adcodes are Chinese administrative division codes
        - Returns Chinese weather descriptions
        - Free tier has rate limits
    """

    def __init__(
        self,
        key: str,
        *,
        timeout: float = 5.0,
        max_retries: int = 3,
        retry_wait: float = 2.0,
    ):
        self.key = key
        super().__init__(base_url="https://restapi.amap.com/v3/weather/", timeout=timeout)
        self.max_retries = max_retries
        self.retry_wait = retry_wait

    async def fetch(self, adcode: str, /) -> WeatherInfo:
        """Fetch current weather for a location by administrative code.

        Args:
            adcode: Chinese administrative division code
                (e.g., "110000" for Beijing, "310000" for Shanghai).

        Returns:
            WeatherInfo with current weather conditions.

        Raises:
            RESTError: If Amap API returns an error or no data is available.

        Example:
            ```python
            # Beijing weather
            weather = await service.fetch("110000")
            print(
                f"{weather['address']}:"
                f"{weather['temperature']}°C, {weather['condition']}"
            )
            # "北京市 北京市: 15°C, 晴"
            ```
        """
        params = AmapWeatherInfoParams(key=self.key, city=adcode, extensions="base", output="JSON")
        response = await self.request(
            endpoint="weatherInfo",
            method="GET",
            params=params,
            cast_to=AmapWeatherInfo,
            max_retries=self.max_retries,
            retry_wait=self.retry_wait,
        )
        if response.status != "1" or response.infocode != "10000":
            logger.error("Amap Weather API error: %s", response.info)
            raise RESTError(f"Failed to fetch weather data: {response.info}")
        if not response.lives or len(response.lives) == 0:
            logger.error("No live weather data available in response")
            raise RESTError("No live weather data available")
        live = response.lives[0]

        logger.debug(
            "Fetched weather data: %s, %s°C, %s, Wind: %s at level %s, Humidity: %s%%, Reported at: %s",
            f"{live['province']} {live['city']}",
            live["weather"],
            live["temperature"],
            f"{live['winddirection']}°",
            live["windpower"],
            live["humidity"],
            live["reporttime"],
        )
        return WeatherInfo(
            address=f"{live['province']} {live['city']}",
            condition=live["weather"],
            temperature=live["temperature"],
            winddirection=live["winddirection"],
            windpower=live["windpower"],
            humidity=live["humidity"],
            reporttime=live["reporttime"],
        )
