"""Tests for rate limiting via slowapi on API endpoints.

Validates per-key rate limits: /submit at 5/min, /status and /result at 30/min.
429 responses must include Retry-After header.

Each test creates a fresh Limiter instance to avoid cross-test state pollution.
"""
import hashlib
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from omniparse.api.rate_limit import _get_api_key, limiter, rate_limit_exceeded_handler


# -------------------------------------------------------------------
# Unit tests: _get_api_key function
# -------------------------------------------------------------------

class _FakeRequest:
    """Minimal mock of starlette Request with headers dict."""
    def __init__(self, headers: dict | None = None):
        self.headers = headers or {}


def test_get_api_key_extracts_header():
    """_get_api_key returns the X-API-Key header value."""
    req = _FakeRequest({"x-api-key": "my-key-42"})
    assert _get_api_key(req) == "my-key-42"


def test_get_api_key_returns_anonymous_when_missing():
    """_get_api_key returns 'anonymous' when X-API-Key header is absent."""
    req = _FakeRequest({})
    assert _get_api_key(req) == "anonymous"


def test_limiter_is_limiter_instance():
    """Module-level limiter is a slowapi Limiter instance."""
    assert isinstance(limiter, Limiter)


# -------------------------------------------------------------------
# Integration tests: rate limiting on endpoints
# -------------------------------------------------------------------

RATE_HEADERS = {"X-Api-Key": "rate-test-key"}
RATE_FILE = ("file", ("test.pdf", b"%PDF-1.4 fake content", "application/pdf"))


def _make_rate_limit_app():
    """Create a FastAPI app with a FRESH limiter + API router for rate limit testing."""
    from omniparse.api.auth import register_api_key

    # Create a FRESH limiter to avoid cross-test state
    fresh_limiter = Limiter(key_func=_get_api_key)

    api_keys_store: dict = {}
    register_api_key(api_keys_store, "rate-test-key", budget_usd=100.0)
    jobs_store: dict = {}

    call_counter = [0]

    def spawn_fn(*args, **kwargs):
        call_counter[0] += 1
        call = MagicMock()
        call.object_id = f"rate_job_{call_counter[0]}"
        return call

    # Patch the module-level limiter so decorators use our fresh one
    with patch("omniparse.api.router.limiter", fresh_limiter):
        from omniparse.api.router import create_api_router
        app = FastAPI()
        app.state.limiter = fresh_limiter
        app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)
        app.add_middleware(SlowAPIMiddleware)
        router = create_api_router(api_keys_store, jobs_store, "test-secret", spawn_fn=spawn_fn)
        app.include_router(router)

    return TestClient(app), api_keys_store, jobs_store


def test_submit_rate_limit_5_per_minute():
    """6th POST /submit within 1 minute from same API key returns 429."""
    client, _, _ = _make_rate_limit_app()

    # First 5 should succeed
    for i in range(5):
        resp = client.post("/submit", headers=RATE_HEADERS, files=[RATE_FILE])
        assert resp.status_code == 200, f"Request {i+1} should succeed, got {resp.status_code}"

    # 6th should be rate-limited
    resp = client.post("/submit", headers=RATE_HEADERS, files=[RATE_FILE])
    assert resp.status_code == 429, f"6th request should be 429, got {resp.status_code}"


def test_submit_429_has_retry_after_header():
    """429 response includes Retry-After header."""
    client, _, _ = _make_rate_limit_app()

    # Exhaust the limit
    for _ in range(5):
        client.post("/submit", headers=RATE_HEADERS, files=[RATE_FILE])

    resp = client.post("/submit", headers=RATE_HEADERS, files=[RATE_FILE])
    assert resp.status_code == 429
    assert "Retry-After" in resp.headers, "429 response must include Retry-After header"


def test_status_rate_limit_30_per_minute():
    """31st GET /status within 1 minute from same API key returns 429."""
    client, _, jobs_store = _make_rate_limit_app()

    job_key_hash = hashlib.sha256("rate-test-key".encode()).hexdigest()
    jobs_store["status_job"] = {
        "api_key_hash": job_key_hash,
        "status": "processing",
    }

    for i in range(30):
        resp = client.get("/status/status_job", headers=RATE_HEADERS)
        assert resp.status_code == 200, f"Request {i+1} should succeed, got {resp.status_code}"

    resp = client.get("/status/status_job", headers=RATE_HEADERS)
    assert resp.status_code == 429, f"31st request should be 429, got {resp.status_code}"


def test_result_rate_limit_30_per_minute():
    """31st GET /result within 1 minute from same API key returns 429."""
    client, _, jobs_store = _make_rate_limit_app()

    job_key_hash = hashlib.sha256("rate-test-key".encode()).hexdigest()
    jobs_store["result_job"] = {
        "api_key_hash": job_key_hash,
        "status": "processing",
    }

    for i in range(30):
        resp = client.get("/result/result_job", headers=RATE_HEADERS)
        assert resp.status_code == 202, f"Request {i+1} should succeed, got {resp.status_code}"

    resp = client.get("/result/result_job", headers=RATE_HEADERS)
    assert resp.status_code == 429, f"31st request should be 429, got {resp.status_code}"
