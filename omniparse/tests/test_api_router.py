"""Tests for the async submission API router.

Uses plain dicts for stores and a mock spawn function to avoid any
Modal dependency.  Validates submit, status, and result endpoints
including budget_usd pass-through, auth failures, ownership verification,
and input validation (SSRF callback URL, file size/magic byte).
"""
import hashlib
import inspect
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from omniparse.api.auth import register_api_key, SPEND_BUCKET_PREFIX
from omniparse.api.router import create_api_router


@pytest.fixture()
def api_keys_store():
    store: dict = {}
    register_api_key(store, "test-key-123", budget_usd=10.0)
    return store


@pytest.fixture()
def jobs_store():
    return {}


@pytest.fixture()
def mock_spawn():
    fn = MagicMock()
    call = MagicMock()
    call.object_id = "job_abc123"
    fn.return_value = call
    return fn


@pytest.fixture(autouse=True)
def _disable_rate_limiter():
    """Disable rate limiting for API router tests -- rate limit behavior tested in test_rate_limit.py."""
    from omniparse.api.rate_limit import limiter
    limiter.enabled = False
    yield
    limiter.enabled = True


@pytest.fixture()
def client(api_keys_store, jobs_store, mock_spawn):
    app = FastAPI()
    router = create_api_router(api_keys_store, jobs_store, "test-secret", spawn_fn=mock_spawn)
    app.include_router(router)
    return TestClient(app)


HEADERS = {"X-Api-Key": "test-key-123"}
FILE = ("file", ("test.pdf", b"%PDF-1.4 fake content", "application/pdf"))


def test_submit_returns_job_id(client):
    resp = client.post("/submit", headers=HEADERS, files=[FILE])
    assert resp.status_code == 200
    body = resp.json()
    assert body["job_id"] == "job_abc123"
    assert body["status"] == "processing"


def test_submit_invalid_api_key(client):
    resp = client.post("/submit", headers={"X-Api-Key": "bad-key"}, files=[FILE])
    assert resp.status_code == 401


def test_submit_budget_exhausted(client, api_keys_store):
    # Exhaust budget by writing a spend bucket key >= budget
    api_keys_store[f"{SPEND_BUCKET_PREFIX}test-key-123:2026010100"] = 10.0
    resp = client.post("/submit", headers=HEADERS, files=[FILE])
    assert resp.status_code == 429


def test_status_processing(client, jobs_store):
    # Submit a job first
    client.post("/submit", headers=HEADERS, files=[FILE])
    resp = client.get("/status/job_abc123", headers=HEADERS)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "processing"
    assert body["job_id"] == "job_abc123"


def test_status_not_found(client):
    resp = client.get("/status/nonexistent", headers=HEADERS)
    assert resp.status_code == 404


def test_result_not_ready(client, jobs_store):
    client.post("/submit", headers=HEADERS, files=[FILE])
    resp = client.get("/result/job_abc123", headers=HEADERS)
    assert resp.status_code == 202
    assert resp.json()["status"] == "processing"


def test_result_completed(client, jobs_store):
    # Directly set completed status with result (include hashed key for ownership)
    jobs_store["job_completed"] = {
        "api_key_hash": hashlib.sha256("test-key-123".encode()).hexdigest(),
        "status": "completed",
        "result": {"markdown": "# Hello", "metadata": {"page_count": 1}},
    }
    resp = client.get("/result/job_completed", headers=HEADERS)
    assert resp.status_code == 200
    body = resp.json()
    assert body["markdown"] == "# Hello"


def test_custom_ce_threshold_in_job_meta(client, jobs_store):
    resp = client.post(
        "/submit",
        headers=HEADERS,
        files=[FILE],
        data={"ce_threshold": "0.8"},
    )
    assert resp.status_code == 200
    meta = jobs_store["job_abc123"]
    assert meta["ce_threshold"] == 0.8


def test_submit_passes_budget_to_spawn(client, mock_spawn):
    client.post("/submit", headers=HEADERS, files=[FILE])
    # spawn_fn should have been called with (file_bytes, filename, budget_usd)
    mock_spawn.assert_called_once()
    args = mock_spawn.call_args
    assert args[0][0] == b"%PDF-1.4 fake content"  # file_bytes
    assert args[0][1] == "test.pdf"  # filename
    assert args[0][2] == 10.0  # budget_usd from the registered API key


def test_submit_stores_budget_in_job_meta(client, jobs_store):
    client.post("/submit", headers=HEADERS, files=[FILE])
    meta = jobs_store["job_abc123"]
    assert meta["budget_usd"] == 10.0


def test_submit_invalid_key_generic_message(client):
    """401 error returns generic 'Invalid API key' detail, not internal exception message."""
    resp = client.post("/submit", headers={"X-Api-Key": "bad-key"}, files=[FILE])
    assert resp.status_code == 401
    body = resp.json()
    assert body["detail"] == "Invalid API key"


def test_submit_budget_exhausted_generic_message(client, api_keys_store):
    """429 error returns generic 'Budget exhausted' detail, not internal exception message."""
    api_keys_store[f"{SPEND_BUCKET_PREFIX}test-key-123:2026010100"] = 10.0
    resp = client.post("/submit", headers=HEADERS, files=[FILE])
    assert resp.status_code == 429
    body = resp.json()
    assert body["detail"] == "Budget exhausted"


# --- Auth and ownership verification tests ---


def test_status_requires_api_key(client, jobs_store):
    """GET /status with no X-Api-Key header returns 401 with 'Missing API key'."""
    client.post("/submit", headers=HEADERS, files=[FILE])
    resp = client.get("/status/job_abc123")
    assert resp.status_code == 401
    assert resp.json()["detail"] == "Missing API key"


def test_status_rejects_wrong_key(client, api_keys_store, jobs_store):
    """Submit with key A, then GET /status with key B returns 403 'Access denied'."""
    register_api_key(api_keys_store, "key-B", budget_usd=10.0)
    client.post("/submit", headers=HEADERS, files=[FILE])
    resp = client.get("/status/job_abc123", headers={"X-Api-Key": "key-B"})
    assert resp.status_code == 403
    assert resp.json()["detail"] == "Access denied"


def test_status_with_valid_owner(client, jobs_store):
    """Submit with key A, then GET /status with key A returns 200 processing."""
    client.post("/submit", headers=HEADERS, files=[FILE])
    resp = client.get("/status/job_abc123", headers=HEADERS)
    assert resp.status_code == 200
    assert resp.json()["status"] == "processing"


def test_result_requires_api_key(client, jobs_store):
    """GET /result with no X-Api-Key header returns 401."""
    client.post("/submit", headers=HEADERS, files=[FILE])
    resp = client.get("/result/job_abc123")
    assert resp.status_code == 401


def test_result_rejects_wrong_key(client, api_keys_store, jobs_store):
    """Submit with key A, then GET /result with key B returns 403."""
    register_api_key(api_keys_store, "key-B", budget_usd=10.0)
    client.post("/submit", headers=HEADERS, files=[FILE])
    resp = client.get("/result/job_abc123", headers={"X-Api-Key": "key-B"})
    assert resp.status_code == 403


def test_submit_stores_hashed_key(client, jobs_store):
    """After POST /submit, jobs_store has api_key_hash (64-char hex), not api_key."""
    client.post("/submit", headers=HEADERS, files=[FILE])
    meta = jobs_store["job_abc123"]
    assert "api_key" not in meta, "Plaintext api_key must not exist in job metadata"
    assert "api_key_hash" in meta
    assert len(meta["api_key_hash"]) == 64  # SHA-256 hex digest


def test_submit_hashed_key_matches_sha256(client, jobs_store):
    """Stored hash matches hashlib.sha256 of the submitted key."""
    client.post("/submit", headers=HEADERS, files=[FILE])
    meta = jobs_store["job_abc123"]
    expected = hashlib.sha256("test-key-123".encode()).hexdigest()
    assert meta["api_key_hash"] == expected


def test_ownership_uses_hmac_compare_digest():
    """Source code of router.py contains hmac.compare_digest for constant-time comparison."""
    import omniparse.api.router as router_mod
    source = inspect.getsource(router_mod)
    assert "hmac.compare_digest" in source


# --- Input validation tests (SSRF-01, INPT-01) ---


def _mock_getaddrinfo_public(host, port, *args, **kwargs):
    """Mock DNS returning a public IP."""
    return [(2, 1, 6, "", ("93.184.216.34", port))]


def _mock_getaddrinfo_private(host, port, *args, **kwargs):
    """Mock DNS returning a private IP."""
    return [(2, 1, 6, "", ("10.0.0.1", port))]


def test_submit_rejects_http_callback(client):
    """POST /submit with http:// callback_url returns 400."""
    resp = client.post(
        "/submit",
        headers=HEADERS,
        files=[FILE],
        data={"callback_url": "http://evil.com/hook"},
    )
    assert resp.status_code == 400
    assert "HTTPS" in resp.json()["detail"]


@patch("omniparse.api.validation.socket.getaddrinfo", side_effect=_mock_getaddrinfo_private)
def test_submit_rejects_private_ip_callback(_mock, client):
    """POST /submit with callback_url resolving to private IP returns 400."""
    resp = client.post(
        "/submit",
        headers=HEADERS,
        files=[FILE],
        data={"callback_url": "https://internal.local/hook"},
    )
    assert resp.status_code == 400
    assert "private" in resp.json()["detail"].lower()


@patch("omniparse.api.validation.socket.getaddrinfo", side_effect=_mock_getaddrinfo_public)
def test_submit_accepts_valid_https_callback(_mock, client):
    """POST /submit with valid HTTPS callback_url returns 200."""
    resp = client.post(
        "/submit",
        headers=HEADERS,
        files=[FILE],
        data={"callback_url": "https://example.com/hook"},
    )
    assert resp.status_code == 200


def test_submit_rejects_fake_pdf(client):
    """POST /submit with .pdf file lacking %PDF- magic returns 422."""
    fake_pdf = ("file", ("test.pdf", b"NOT-A-PDF-FILE", "application/pdf"))
    resp = client.post("/submit", headers=HEADERS, files=[fake_pdf])
    assert resp.status_code == 422
    assert "PDF signature" in resp.json()["detail"]


def test_submit_accepts_valid_pdf(client):
    """POST /submit with valid PDF magic bytes returns 200."""
    valid_pdf = ("file", ("test.pdf", b"%PDF-1.4 content here", "application/pdf"))
    resp = client.post("/submit", headers=HEADERS, files=[valid_pdf])
    assert resp.status_code == 200


def test_submit_validation_before_spawn(client, mock_spawn):
    """After callback URL rejection, spawn_fn must NOT have been called."""
    resp = client.post(
        "/submit",
        headers=HEADERS,
        files=[FILE],
        data={"callback_url": "http://evil.com/hook"},
    )
    assert resp.status_code == 400
    mock_spawn.assert_not_called()


def test_submit_rejects_oversized_file(client):
    """POST /submit with Content-Length > 100MB returns 413."""
    # Use a small actual file but set Content-Length header to > 100MB
    # The pre-check should catch this before reading the body
    big_size = 100 * 1024 * 1024 + 1
    big_bytes = b"%PDF-" + b"\x00" * (big_size - 5)
    oversized_pdf = ("file", ("big.pdf", big_bytes, "application/pdf"))
    resp = client.post("/submit", headers=HEADERS, files=[oversized_pdf])
    assert resp.status_code == 413


def test_submit_accepts_non_pdf_without_magic(client):
    """POST /submit with non-PDF file skips magic byte check."""
    png_file = ("file", ("image.png", b"PNG-file-bytes", "image/png"))
    resp = client.post("/submit", headers=HEADERS, files=[png_file])
    assert resp.status_code == 200


def test_submit_has_request_parameter():
    """Router source uses request: Request for Content-Length access."""
    import omniparse.api.router as router_mod
    source = inspect.getsource(router_mod)
    assert "request: Request" in source
    assert "content-length" in source


def test_submit_stores_created_at(client, jobs_store):
    """After POST /submit, job metadata includes created_at as a float epoch timestamp."""
    client.post("/submit", headers=HEADERS, files=[FILE])
    meta = jobs_store["job_abc123"]
    assert "created_at" in meta
    assert isinstance(meta["created_at"], float)
