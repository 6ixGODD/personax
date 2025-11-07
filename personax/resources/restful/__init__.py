from __future__ import annotations

import abc
import typing as t

import httpx
import pydantic as pydt
import tenacity

from personax.utils import AsyncContextMixin

RespT = t.TypeVar("RespT", bound=pydt.BaseModel)


class BearerAuth(httpx.Auth):
    """HTTP Bearer token authentication handler for httpx.

    Implements the httpx.Auth interface to automatically add Bearer token
    authentication to requests.

    Args:
        token: The bearer token to use for authentication.
    """

    def __init__(self, token: str):
        self.token = token

    def auth_flow(self, request: httpx.Request) -> t.Generator[httpx.Request, httpx.Response, None]:
        """Add Bearer token to request Authorization header.

        Args:
            request: The HTTP request to authenticate.

        Yields:
            The authenticated request with Authorization header.
        """
        request.headers["Authorization"] = f"Bearer {self.token}"
        yield request


class RESTfulService(AsyncContextMixin, abc.ABC):
    """Abstract base class for RESTful API service clients.

    Provides a high-level interface for making HTTP requests to RESTful APIs
    with automatic retry logic, response validation using Pydantic models,
    and built-in error handling.

    Features:
    - Automatic retry with configurable attempts and wait time
    - Response validation and parsing using Pydantic models
    - Support for Bearer token and custom authentication
    - Proxy support
    - Configurable timeouts and default headers/params
    - Async context manager support

    Attributes:
        url: The base URL for the API.
        http_client: The underlying httpx AsyncClient instance.

    Args:
        base_url: Base URL for all API requests.
        auth: Optional httpx authentication handler.
        proxy: Optional proxy URL.
        timeout: Request timeout in seconds. Defaults to 10.0.
        http_client: Optional pre-configured httpx.AsyncClient. If not provided,
            a new client will be created.
        default_headers: Optional default headers for all requests.
        default_params: Optional default query parameters for all requests.

    Example:
        ```python
        class MyAPIResponse(pydt.BaseModel):
            id: str
            name: str


        class MyAPI(RESTfulService):
            async def get_item(self, item_id: str) -> MyAPIResponse:
                return await self.request(
                    f"/items/{item_id}",
                    method="GET",
                    cast_to=MyAPIResponse,
                )


        async with MyAPI(
            "https://api.example.com", auth=BearerAuth("token")
        ) as api:
            item = await api.get_item("123")
            print(item.name)
        ```
    """
    def __init__(
        self,
        base_url: str,
        *,
        auth: httpx.Auth | None = None,
        proxy: str | httpx.URL | None = None,
        timeout: float = 10.0,
        http_client: httpx.AsyncClient | None = None,
        default_headers: dict[str, str] | None = None,
        default_params: dict[str, t.Any] | None = None,
    ):
        self.url = base_url
        self.http_client = http_client or httpx.AsyncClient(
            base_url=base_url,
            auth=auth,
            proxy=proxy,
            timeout=timeout,
            headers=default_headers,
            params=default_params,
        )

    async def request(
        self,
        endpoint: str,
        *,
        method: t.Literal["GET", "POST", "PUT", "DELETE"] = "GET",
        headers: t.Mapping[str, str] | None = None,
        params: t.Mapping[str, t.Any] | None = None,
        json: t.Mapping[str, t.Any] | None = None,
        cast_to: type[RespT],
        max_retries: int = 3,
        retry_wait: float = 2.0,
    ) -> RespT:
        """Make an HTTP request with automatic retry and response validation.

        Sends an HTTP request to the specified endpoint and parses the response
        into the specified Pydantic model. Automatically retries on failure and
        validates the response data.

        Args:
            endpoint: API endpoint path (relative to base_url).
            method: HTTP method to use. Defaults to "GET".
            headers: Optional request headers to merge with default headers.
            params: Optional query parameters to merge with default params.
            json: Optional JSON body for POST/PUT requests.
            cast_to: Pydantic model class to parse the response into.
            max_retries: Maximum number of retry attempts. Defaults to 3.
            retry_wait: Wait time in seconds between retries. Defaults to 2.0.

        Returns:
            Parsed response as an instance of cast_to model.

        Raises:
            httpx.HTTPStatusError: If the response status indicates an error
                after all retries.
            pydt.ValidationError: If response validation fails.
            tenacity.RetryError: If all retry attempts are exhausted.

        Example:
            ```python
            class UserResponse(pydt.BaseModel):
                user_id: str
                username: str


            user = await service.request(
                "/users/123",
                method="GET",
                cast_to=UserResponse,
                max_retries=5,
            )
            ```
        """
        @tenacity.retry(
            stop=tenacity.stop_after_attempt(max_retries),
            wait=tenacity.wait_fixed(retry_wait),
            reraise=True,
        )
        async def _make_request(
            method: str,
            endpoint: str,
            headers: t.Mapping[str, str] | None,
            params: t.Mapping[str, t.Any] | None,
            json: t.Mapping[str, t.Any] | None,
        ) -> httpx.Response:
            response = await self.http_client.request(
                method, endpoint, headers=headers, params=params, json=json
            )
            response.raise_for_status()
            return response

        response = await _make_request(method, endpoint, headers, params, json)
        return cast_to.model_validate(response.json())

    async def close(self) -> None:
        await self.http_client.aclose()
