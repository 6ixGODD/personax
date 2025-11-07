from __future__ import annotations

import abc

import typing_extensions as te

from personax.resources.restful import RESTfulService


class WeatherInfo(te.TypedDict, total=False):
    """Weather information data structure.

    Contains current weather conditions for a specific location.
    """

    address: te.Required[str]
    """Human-readable location address."""

    condition: te.Required[str]
    """Current weather condition description (e.g., "Sunny", "Rainy")."""

    temperature: te.Required[str]
    """Current temperature in Celsius."""

    winddirection: str
    """Wind direction (e.g., "NW", "180°")."""

    windpower: str
    """Wind speed or power level."""

    humidity: str
    """Humidity percentage."""

    reporttime: te.Required[str]
    """Timestamp when weather data was reported."""


class WeatherInfoService(RESTfulService):
    """Abstract base class for weather information services.

    Defines the interface for weather data providers. Implementations
    should integrate with specific weather APIs (e.g., Amap, OpenWeatherMap).

    Example:
        ```python
        class CustomWeatherService(WeatherInfoService):
            async def fetch(self, adcode: str) -> WeatherInfo:
                response = await self.request(
                    f"/weather/{adcode}",
                    method="GET",
                    cast_to=CustomWeatherResponse,
                )
                return WeatherInfo(
                    address=response.location,
                    temperature=str(response.temp),
                    condition=response.weather,
                    reporttime=response.timestamp,
                )
        ```
    """

    @abc.abstractmethod
    async def fetch(self, adcode: str, /) -> WeatherInfo:
        """Fetch current weather information for a location.

        Args:
            adcode: Administrative division code identifying the location
                (e.g., "110000" for Beijing in China).

        Returns:
            WeatherInfo containing current weather data.

        Raises:
            RESTError: If the weather service request fails.
        """
