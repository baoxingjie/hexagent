"""DingTalk access token management with in-memory caching."""

from __future__ import annotations

import logging
import time

import httpx

logger = logging.getLogger(__name__)

# (client_id, client_secret) -> (access_token, expire_at_unix_seconds)
_TOKEN_CACHE: dict[tuple[str, str], tuple[str, float]] = {}

_DINGTALK_TOKEN_URL = "https://api.dingtalk.com/v1.0/oauth2/accessToken"
_EXPIRE_BUFFER_SECS = 60  # refresh this many seconds before actual expiry


async def get_access_token(client_id: str, client_secret: str) -> str:
    """Return a valid DingTalk access token, fetching a new one when needed.

    Tokens are cached in memory; a new token is fetched when the cached one
    is within ``_EXPIRE_BUFFER_SECS`` of expiry.

    Args:
        client_id: DingTalk app key.
        client_secret: DingTalk app secret.

    Returns:
        A valid access token string.

    Raises:
        httpx.HTTPStatusError: If the DingTalk API returns a non-2xx response.
    """
    cache_key = (client_id, client_secret)
    cached = _TOKEN_CACHE.get(cache_key)
    if cached is not None:
        token, expire_at = cached
        if time.time() < expire_at - _EXPIRE_BUFFER_SECS:
            return token

    logger.debug("Fetching new DingTalk access token for clientId=%s", client_id)
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(
            _DINGTALK_TOKEN_URL,
            json={"appKey": client_id, "appSecret": client_secret},
        )
        resp.raise_for_status()
        data = resp.json()

    token: str = data["accessToken"]
    expire_in: int = data.get("expireIn", 7200)
    _TOKEN_CACHE[cache_key] = (token, time.time() + expire_in)
    logger.debug("Access token obtained, expires in %ds", expire_in)
    return token


def invalidate_token(client_id: str, client_secret: str) -> None:
    """Remove a cached token so the next call fetches a fresh one."""
    _TOKEN_CACHE.pop((client_id, client_secret), None)
