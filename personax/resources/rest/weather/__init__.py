from __future__ import annotations

import abc

import typing_extensions as te

from personax.resources.rest import RESTResource


class WeatherInfo(te.TypedDict, total=False):
    address: te.Required[str]
    """Human-readable address."""

    condition: te.Required[str]
    """Current weather condition description."""

    temperature: te.Required[str]
    """Current temperature in Celsius."""

    winddirection: str
    """Wind direction."""

    windpower: str
    """Wind power level."""

    humidity: str
    """Humidity percentage."""

    reporttime: te.Required[str]
    """Data report time."""


class WeatherInfoService(RESTResource):
    @abc.abstractmethod
    async def fetch(self, adcode: str, /) -> WeatherInfo:
        pass
