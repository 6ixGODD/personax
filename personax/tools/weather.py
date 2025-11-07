from __future__ import annotations

import logging
import typing as t

import typing_extensions as te

from personax.exceptions import RESTError
from personax.exceptions import ToolCallError
from personax.resources.restful.weather import WeatherInfoService
from personax.tools import BaseTool
from personax.tools import Property

logger = logging.getLogger("personax.tools.weather")


class Weather(te.TypedDict, total=False):
    """Weather information result structure.

    Contains current weather data for a specific location.
    """

    location: te.Required[str]
    """Human-readable location name."""

    temperature: te.Required[str]
    """Current temperature in Celsius."""

    condition: str
    """Weather condition description (e.g., "Sunny", "Rainy")."""

    humidity: str
    """Current humidity percentage."""

    windpower: str
    """Wind speed and direction description."""


class GetWeather(BaseTool[[str], Weather]):
    """Tool for retrieving current weather information.

    Fetches weather data for a location specified by its administrative code
    using an external weather information service. Designed for use with
    LLMs that need real-time weather context.

    The tool uses administrative codes (adcodes) to identify locations, which
    are standardized geographic identifiers used in many regional weather APIs.

    Attributes:
        __function_description__: Tool description for LLM function calling.
        weather_srv: Weather information service client.

    Args:
        weather_srv: Configured weather service for data retrieval.

    Example:
        ```python
        # Initialize with weather service
        from personax.resources.restful.weather.amap import (
            AmapWeatherInfoService,
        )

        weather_service = AmapWeatherInfoService(key="YOUR_API_KEY")
        weather_tool = GetWeather(weather_srv=weather_service)

        # Use directly
        weather = await weather_tool(adcode="110000")  # Beijing
        print(weather["location"])  # "北京市"
        print(weather["temperature"])  # "15"
        print(weather["condition"])  # "晴"


        # Use with CompletionSystem
        completion = await system.complete(
            messages=Messages(
                messages=[
                    Message(
                        role="system",
                        content="You are a helpful assistant.",
                    ),
                    Message(
                        role="user",
                        content="What's the weather in Beijing?",
                    ),
                ]
            ),
            model="gpt-4",
            tools=[weather_tool],
        )
        # LLM will call get_weather(adcode="110000") and use result in response


        # Schema inspection
        print(weather_tool.schema_json)
        # {
        #   "type": "function",
        #   "function": {
        #     "name": "get_weather",
        #     "description": "Get the current weather information...",
        #     "parameters": {
        #       "type": "object",
        #       "properties": {
        #         "adcode": {
        #           "type": "string",
        #           "description": "The administrative code...",
        #           "example": "110000"
        #         }
        #       },
        #       "required": ["adcode"]
        #     }
        #   }
        # }
        ```

    Raises:
        ToolCallError: If weather service fails to fetch data.

    Note:
        - Requires a configured WeatherInfoService instance
        - Administrative codes are region-specific (e.g., Chinese adcodes)
        - Weather service errors are wrapped in ToolCallError for LLM handling
    """

    __function_description__ = (
        "Get the current weather information for a given location by its administrative code."
    )

    def __init__(self, weather_srv: WeatherInfoService):
        self.weather_srv = weather_srv

    async def __call__(
        self,
        adcode: t.Annotated[
            str,
            Property(
                description="The administrative code of the location to get the weather for.",
                examples=["110000"],
            ),
        ],
    ) -> Weather:
        """Fetch weather information for a location.

        Args:
            adcode: Administrative code identifying the location
                (e.g., "110000" for Beijing).

        Returns:
            Weather data including location, temperature, condition,
            humidity, and wind information.

        Raises:
            ToolCallError: If the weather service fails to retrieve data.

        Example:
            ```python
            weather = await tool(adcode="310000")  # Shanghai
            print(f"{weather['location']}: {weather['temperature']}°C")
            ```
        """
        logger.debug("Fetching weather info for adcode: %s", adcode)
        try:
            info = await self.weather_srv.fetch(adcode)
        except RESTError as exc:
            logger.error("Error fetching weather info for adcode %s: %s", adcode, exc)
            raise ToolCallError(f"Failed to get weather info for adcode {adcode}: {exc}") from exc

        logger.debug("Weather info retrieved: %s", info)
        return Weather(
            location=info["address"],
            temperature=info["temperature"],
            condition=info.get("condition", ""),
            humidity=info.get("humidity", ""),
            windpower=info.get("windpower", ""),
        )
