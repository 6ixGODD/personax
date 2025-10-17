from __future__ import annotations

import typing as t

import pydantic as pydt
import typing_extensions as te

from personax.exceptions import RESTResourceException
from personax.resources.rest.weather import WeatherInfo
from personax.resources.rest.weather import WeatherInfoService


class AmapWeatherInfoParams(te.TypedDict):
    key: str
    """API key for Amap Weather Service"""

    city: str
    """Adcode of the city, e.g., '110000' for Beijing"""

    extensions: t.Literal["base"]
    """Type of weather data: 'base' for current weather"""

    output: t.Literal["JSON"]
    """Response format: 'JSON'"""


class AmapWeatherInfoLives(te.TypedDict):
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
    status: str = pydt.Field(..., description='Response status, "1" indicates success')

    count: str | None = pydt.Field(None, description="Number of results returned")

    info: str | None = pydt.Field(None, description='Response information, "OK" indicates success')

    infocode: str | None = pydt.Field(None, description='Response code, "10000" indicates success')

    lives: t.List[AmapWeatherInfoLives] = pydt.Field(
        default_factory=list,
        description="List of live weather information",
    )


class AmapWeatherInfoService(WeatherInfoService):

    def __init__(
        self, key: str, *, timeout: float = 5.0, max_retries: int = 3, retry_wait: float = 2.0
    ):
        self.key = key
        super().__init__(base_url="https://restapi.amap.com/v3/weather/", timeout=timeout)
        self.max_retries = max_retries
        self.retry_wait = retry_wait

    async def fetch(self, adcode: str, /) -> WeatherInfo:
        params = AmapWeatherInfoParams(key=self.key, city=adcode, extensions="base", output="JSON")
        response = await self.request(
            endpoint="weatherInfo",
            method="GET",
            params=params,
            cast_to=AmapWeatherInfo,
            max_retries=self.max_retries,
            retry_wait=self.retry_wait
        )
        if response.status != "1" or response.infocode != "10000":
            raise RESTResourceException(f"Failed to fetch weather data: {response.info}")
        if not response.lives or len(response.lives) == 0:
            raise RESTResourceException("No live weather data available")
        live = response.lives[0]

        return WeatherInfo(
            address=f'{live["province"]} {live["city"]}',
            condition=live["weather"],
            temperature=live["temperature"],
            winddirection=live["winddirection"],
            windpower=live["windpower"],
            humidity=live["humidity"],
            reporttime=live["reporttime"]
        )
