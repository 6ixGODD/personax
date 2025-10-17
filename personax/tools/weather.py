from __future__ import annotations

import typing as t

import typing_extensions as te

from personax.exceptions import RESTResourceException, ToolCallException
from personax.resources.rest.weather import WeatherInfoService
from personax.tools import BaseTool
from personax.tools import Property


class Weather(te.TypedDict, total=False):
    location: te.Required[str]
    """Location name, e.g., "San Francisco, CA, USA"."""

    temperature: te.Required[str]
    """Current temperature in Celsius."""

    condition: str
    """Weather condition description, e.g., "Sunny", "Cloudy"."""

    humidity: str
    """Current humidity percentage."""

    windpower: str
    """Wind power description, e.g., "5 km/h NW"."""


class GetWeather(BaseTool[[str], Weather]):
    __function_description__ = (
        "Get the current weather information for a "
        "given location by its administrative code."
    )

    def __init__(self, weather_srv: WeatherInfoService):
        self.weather_srv = weather_srv

    async def __call__(
        self,
        adcode: t.Annotated[
            str,
            Property(
                description="The administrative code of the location to get "
                "the weather for.",
                example="110000"
            ),
        ],
    ) -> Weather:
        try:
            info = await self.weather_srv.fetch(adcode)
        except RESTResourceException as exc:
            raise ToolCallException(
                f"Failed to get weather info for adcode {adcode}: {exc}"
            ) from exc
        return Weather(
            location=info["address"],
            temperature=info["temperature"],
            condition=info.get("condition", ""),
            humidity=info.get("humidity", ""),
            windpower=info.get("windpower", "")
        )
