"""Webhook delivery -- HMAC-SHA256 signed payloads with exponential backoff retry.

Delivers job completion/failure notifications to caller-specified callback URLs.
Payloads are signed with HMAC-SHA256 so receivers can verify authenticity.
Retries up to 3 times with exponential backoff (1s, 4s, 16s) on non-2xx responses.
"""
import asyncio
import hashlib
import hmac
import json
import logging

import httpx

logger = logging.getLogger(__name__)

# Exponential backoff delays in seconds (3 attempts total)
RETRY_DELAYS = [1.0, 4.0, 16.0]


def sign_payload(body: str, signing_secret: str) -> str:
    """Compute HMAC-SHA256 hex digest of body using signing_secret.

    Args:
        body: JSON string payload to sign.
        signing_secret: Shared secret for HMAC computation.

    Returns:
        Hex-encoded HMAC-SHA256 signature string.
    """
    return hmac.new(
        signing_secret.encode(), body.encode(), hashlib.sha256
    ).hexdigest()


async def deliver_webhook(
    callback_url: str, payload: dict, signing_secret: str
) -> bool:
    """Deliver a signed webhook payload with exponential backoff retry.

    Serializes payload to JSON, signs with HMAC-SHA256, and POSTs to callback_url.
    Retries up to 3 times on non-success responses.

    Args:
        callback_url: URL to POST the webhook to.
        payload: Dict to serialize and deliver.
        signing_secret: Shared secret for HMAC signing.

    Returns:
        True if any attempt received a 2xx/3xx response, False if all retries exhausted.
    """
    body = json.dumps(payload, default=str)
    signature = sign_payload(body, signing_secret)
    headers = {
        "Content-Type": "application/json",
        "X-OmniParse-Signature": signature,
    }

    for i, delay in enumerate(RETRY_DELAYS):
        try:
            async with httpx.AsyncClient(timeout=10.0, follow_redirects=False) as client:
                response = await client.post(
                    callback_url, content=body, headers=headers
                )
                if response.status_code < 400:
                    return True
                logger.warning(
                    "Webhook attempt %d/%d failed: HTTP %d",
                    i + 1,
                    len(RETRY_DELAYS),
                    response.status_code,
                )
        except Exception as exc:
            logger.warning(
                "Webhook attempt %d/%d error: %s", i + 1, len(RETRY_DELAYS), exc
            )

        if i < len(RETRY_DELAYS) - 1:
            await asyncio.sleep(delay)

    logger.error("All webhook retries exhausted for %s", callback_url)
    return False
