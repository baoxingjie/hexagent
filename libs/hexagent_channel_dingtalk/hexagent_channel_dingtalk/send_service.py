"""Send messages to DingTalk conversations via the proactive-messaging API."""

from __future__ import annotations

import json
import logging

import httpx

from hexagent_channel_dingtalk.auth import get_access_token, invalidate_token
from hexagent_channel_dingtalk.config import DingTalkConfig

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.dingtalk.com/v1.0/robot"
_PRIVATE_SEND_URL = f"{_BASE_URL}/oToMessages/batchSend"
_GROUP_SEND_URL = f"{_BASE_URL}/groupMessages/send"


def _auth_headers(token: str) -> dict[str, str]:
    return {
        "x-acs-dingtalk-access-token": token,
        "Content-Type": "application/json",
    }


async def _post_with_token_refresh(
    config: DingTalkConfig,
    url: str,
    payload: dict,
) -> dict:
    """POST ``payload`` to ``url``, retrying once with a fresh token on 401."""
    token = await get_access_token(config.client_id, config.client_secret)
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(url, json=payload, headers=_auth_headers(token))
        if resp.status_code == 401:  # noqa: PLR2004
            # Token may have been revoked server-side; invalidate and retry.
            invalidate_token(config.client_id, config.client_secret)
            token = await get_access_token(config.client_id, config.client_secret)
            resp = await client.post(url, json=payload, headers=_auth_headers(token))
        resp.raise_for_status()
        return resp.json()


async def send_text_to_private(
    config: DingTalkConfig,
    staff_id: str,
    text: str,
) -> dict:
    """Send a plain-text message to a single user in a 1:1 conversation.

    Args:
        config: DingTalk channel configuration.
        staff_id: The recipient's staffId (enterprise user) or openId.
        text: Message content.

    Returns:
        The parsed JSON response from the DingTalk API.
    """
    payload = {
        "robotCode": config.effective_robot_code,
        "userIds": [staff_id],
        "msgParam": json.dumps({"content": text}),
        "msgKey": "sampleText",
    }
    logger.debug("send_text_to_private: staffId=%s, len=%d", staff_id, len(text))
    return await _post_with_token_refresh(config, _PRIVATE_SEND_URL, payload)


async def send_text_to_group(
    config: DingTalkConfig,
    open_conversation_id: str,
    text: str,
) -> dict:
    """Send a plain-text message to a group conversation.

    Args:
        config: DingTalk channel configuration.
        open_conversation_id: The group's openConversationId.
        text: Message content.

    Returns:
        The parsed JSON response from the DingTalk API.
    """
    payload = {
        "robotCode": config.effective_robot_code,
        "openConversationId": open_conversation_id,
        "msgParam": json.dumps({"content": text}),
        "msgKey": "sampleText",
    }
    logger.debug(
        "send_text_to_group: convId=%s, len=%d", open_conversation_id, len(text)
    )
    return await _post_with_token_refresh(config, _GROUP_SEND_URL, payload)


async def send_markdown_to_private(
    config: DingTalkConfig,
    staff_id: str,
    title: str,
    text: str,
) -> dict:
    """Send a Markdown message to a single user in a 1:1 conversation."""
    payload = {
        "robotCode": config.effective_robot_code,
        "userIds": [staff_id],
        "msgParam": json.dumps({"title": title, "text": text}),
        "msgKey": "sampleMarkdown",
    }
    return await _post_with_token_refresh(config, _PRIVATE_SEND_URL, payload)


async def send_markdown_to_group(
    config: DingTalkConfig,
    open_conversation_id: str,
    title: str,
    text: str,
) -> dict:
    """Send a Markdown message to a group conversation."""
    payload = {
        "robotCode": config.effective_robot_code,
        "openConversationId": open_conversation_id,
        "msgParam": json.dumps({"title": title, "text": text}),
        "msgKey": "sampleMarkdown",
    }
    return await _post_with_token_refresh(config, _GROUP_SEND_URL, payload)
