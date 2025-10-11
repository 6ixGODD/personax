from __future__ import annotations

import abc
import typing as t

import httpx
import pydantic as pydt
import tenacity

from personax.helpers.mixin import AsyncContextMixin

RespT = t.TypeVar("RespT", bound=pydt.BaseModel)


class BearerAuth(httpx.Auth):

    def __init__(self, token: str):
        self.token = token

    def auth_flow(self, request: httpx.Request) -> t.Generator[httpx.Request, httpx.Response, None]:
        request.headers["Authorization"] = f"Bearer {self.token}"
        yield request


class RESTResource(abc.ABC, AsyncContextMixin):

    def __init__(self,
                 base_url: str,
                 *,
                 auth: httpx.Auth | None = None,
                 proxy: str | httpx.URL | None = None,
                 timeout: float = 10.0,
                 http_client: httpx.AsyncClient | None = None,
                 default_headers: dict[str, str] | None = None,
                 default_params: dict[str, t.Any] | None = None):
        self.url = base_url
        self.http_client = http_client or httpx.AsyncClient(base_url=base_url,
                                                            auth=auth,
                                                            proxy=proxy,
                                                            timeout=timeout,
                                                            headers=default_headers,
                                                            params=default_params)

    async def request(self,
                      endpoint: str,
                      *,
                      method: t.Literal["GET", "POST", "PUT", "DELETE"] = "GET",
                      headers: t.Mapping[str, str] | None = None,
                      params: t.Mapping[str, t.Any] | None = None,
                      json: t.Mapping[str, t.Any] | None = None,
                      cast_to: t.Type[RespT],
                      max_retries: int = 3,
                      retry_wait: float = 2.0) -> RespT:

        @tenacity.retry(stop=tenacity.stop_after_attempt(max_retries),
                        wait=tenacity.wait_fixed(retry_wait),
                        reraise=True)
        async def _make_request(method: str, endpoint: str, headers: t.Mapping[str, str] | None,
                                params: t.Mapping[str, t.Any] | None,
                                json: t.Mapping[str, t.Any] | None) -> httpx.Response:
            response = await self.http_client.request(method,
                                                      endpoint,
                                                      headers=headers,
                                                      params=params,
                                                      json=json)
            response.raise_for_status()
            return response

        response = await _make_request(method, endpoint, headers, params, json)
        return cast_to.model_validate(response.json())

    async def close(self) -> None:
        await self.http_client.aclose()
