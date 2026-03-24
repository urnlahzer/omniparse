"""Tests for webhook delivery -- HMAC signing, retry logic, header verification."""
import asyncio
import hashlib
import hmac
import json
from unittest.mock import AsyncMock, MagicMock, patch

from omniparse.api.webhooks import sign_payload, deliver_webhook, RETRY_DELAYS


def test_sign_payload_deterministic():
    """Same inputs always produce the same HMAC-SHA256 hex string."""
    body = '{"test":true}'
    secret = "secret123"
    result = sign_payload(body, secret)
    expected = hmac.new(b"secret123", b'{"test":true}', hashlib.sha256).hexdigest()
    assert result == expected
    # Verify determinism
    assert sign_payload(body, secret) == result


def _make_mock_response(status_code: int):
    """Create a mock httpx response with the given status code."""
    resp = MagicMock()
    resp.status_code = status_code
    return resp


def test_webhook_delivery_success():
    """Successful delivery on first attempt returns True, no retries."""
    mock_post = AsyncMock(return_value=_make_mock_response(200))

    with patch("omniparse.api.webhooks.httpx.AsyncClient") as MockClient:
        mock_client_instance = AsyncMock()
        mock_client_instance.post = mock_post
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = mock_client_instance

        result = asyncio.run(
            deliver_webhook("https://example.com/hook", {"job": "1"}, "secret")
        )

    assert result is True
    assert mock_post.call_count == 1


def test_webhook_retry_on_failure():
    """Retries on 500 errors, succeeds on third attempt."""
    mock_post = AsyncMock(
        side_effect=[
            _make_mock_response(500),
            _make_mock_response(500),
            _make_mock_response(200),
        ]
    )

    with patch("omniparse.api.webhooks.httpx.AsyncClient") as MockClient:
        mock_client_instance = AsyncMock()
        mock_client_instance.post = mock_post
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = mock_client_instance

        with patch("omniparse.api.webhooks.asyncio.sleep", new_callable=AsyncMock):
            result = asyncio.run(
                deliver_webhook("https://example.com/hook", {"job": "1"}, "secret")
            )

    assert result is True
    assert mock_post.call_count == 3


def test_webhook_all_retries_exhausted():
    """Returns False after all retry attempts fail."""
    mock_post = AsyncMock(return_value=_make_mock_response(500))

    with patch("omniparse.api.webhooks.httpx.AsyncClient") as MockClient:
        mock_client_instance = AsyncMock()
        mock_client_instance.post = mock_post
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = mock_client_instance

        with patch("omniparse.api.webhooks.asyncio.sleep", new_callable=AsyncMock):
            result = asyncio.run(
                deliver_webhook("https://example.com/hook", {"job": "1"}, "secret")
            )

    assert result is False
    assert mock_post.call_count == 3


def test_webhook_signature_header():
    """X-OmniParse-Signature header is present and matches sign_payload output."""
    mock_post = AsyncMock(return_value=_make_mock_response(200))

    with patch("omniparse.api.webhooks.httpx.AsyncClient") as MockClient:
        mock_client_instance = AsyncMock()
        mock_client_instance.post = mock_post
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = mock_client_instance

        payload = {"job_id": "test-123", "status": "completed"}
        secret = "my-signing-secret"

        result = asyncio.run(
            deliver_webhook("https://example.com/hook", payload, secret)
        )

    assert result is True

    # Extract the headers from the post call
    call_kwargs = mock_post.call_args
    headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers")
    body = call_kwargs.kwargs.get("content") or call_kwargs[1].get("content")

    assert "X-OmniParse-Signature" in headers
    expected_sig = sign_payload(body, secret)
    assert headers["X-OmniParse-Signature"] == expected_sig


# -------------------------------------------------------------------
# Webhook secret enforcement tests (SECR-01)
# -------------------------------------------------------------------

def test_webhook_secret_no_default_fallback():
    """pipeline.py must not contain a default-secret fallback for OMNIPARSE_WEBHOOK_SECRET."""
    import pathlib
    source = pathlib.Path("omniparse/pipeline.py").read_text()
    assert '"default-secret"' not in source, "default-secret fallback must be removed"
    assert "OMNIPARSE_WEBHOOK_SECRET" in source


def test_notify_raises_without_secret():
    """pipeline.py must raise RuntimeError when OMNIPARSE_WEBHOOK_SECRET is unset."""
    import pathlib
    source = pathlib.Path("omniparse/pipeline.py").read_text()
    assert 'raise RuntimeError("OMNIPARSE_WEBHOOK_SECRET must be set")' in source


# -------------------------------------------------------------------
# Webhook redirect prevention (D-03)
# -------------------------------------------------------------------

def test_webhook_no_redirect_follow():
    """httpx.AsyncClient must be created with follow_redirects=False (D-03)."""
    import inspect
    import omniparse.api.webhooks as webhooks_mod
    source = inspect.getsource(webhooks_mod)
    assert "follow_redirects=False" in source
