"""Tests for the HITL review interface -- models, router, and templates.

Uses FastAPI TestClient with an in-memory dict store (no Modal dependency).
"""
import re

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from omniparse.hitl.models import CorrectionPayload, HitlRegion, ReviewItem
from omniparse.hitl.router import create_hitl_router


def _make_test_app(store: dict) -> TestClient:
    """Create a FastAPI app with the HITL router mounted and return a TestClient."""
    from fastapi_csrf_protect.exceptions import CsrfProtectError

    app = FastAPI()
    app.add_exception_handler(
        CsrfProtectError,
        lambda req, exc: JSONResponse(status_code=403, content={"detail": "CSRF validation failed"}),
    )
    router = create_hitl_router(store)
    app.include_router(router)
    return TestClient(app)


def _sample_region(region_id: str = "r1", confidence: float = 0.3) -> dict:
    """Return a minimal HitlRegion dict for test fixtures."""
    return HitlRegion(
        region_id=region_id,
        element_type="text",
        bounding_box=[10.0, 20.0, 100.0, 50.0],
        engine_texts={"pdfplumber": "text A", "paddleocr": "text B"},
        consensus_text="text A",
        confidence=confidence,
    ).model_dump()


def _sample_job(job_id: str = "job123") -> dict:
    """Return a minimal ReviewItem dict for test fixtures."""
    return ReviewItem(
        job_id=job_id,
        filename="test.pdf",
        submitted_at="2026-03-20T12:00:00Z",
        total_regions=5,
        hitl_region_count=2,
        status="pending",
    ).model_dump()


def _submit_with_csrf(client: TestClient, job_id: str, corrections: dict, cookies: dict | None = None) -> "Response":
    """Helper: GET review page to extract CSRF token, then POST with token.

    Performs the full CSRF double-submit flow:
    1. GET /review/{job_id} to extract csrf-token meta tag and csrf cookie
    2. POST /review/{job_id}/submit with X-CSRF-Token header and csrf cookie
    """
    get_cookies = cookies or {}
    resp = client.get(f"/review/{job_id}", cookies=get_cookies)
    assert resp.status_code == 200, f"GET review page failed: {resp.status_code}"

    # Extract CSRF token from meta tag
    match = re.search(r'<meta name="csrf-token" content="([^"]+)"', resp.text)
    assert match, "CSRF meta tag not found in review detail page"
    csrf_token = match.group(1)

    # Extract CSRF cookie from response
    all_cookies = dict(get_cookies)  # Start with provided cookies
    for cookie_header in resp.headers.get_list("set-cookie"):
        if "fastapi-csrf-token" in cookie_header:
            cookie_val = cookie_header.split("=", 1)[1].split(";")[0]
            all_cookies["fastapi-csrf-token"] = cookie_val

    return client.post(
        f"/review/{job_id}/submit",
        json={"corrections": corrections},
        headers={"X-CSRF-Token": csrf_token},
        cookies=all_cookies,
    )


# -------------------------------------------------------------------
# Review list tests
# -------------------------------------------------------------------

def test_review_list_empty():
    """Empty store returns 200 with 'No jobs' message."""
    client = _make_test_app({})
    resp = client.get("/review")
    assert resp.status_code == 200
    assert "No jobs" in resp.text


def test_review_list_renders():
    """Store with a job renders the job_id in the list."""
    store = {"hitl:jobs": [_sample_job("job456")]}
    client = _make_test_app(store)
    resp = client.get("/review")
    assert resp.status_code == 200
    assert "job456" in resp.text
    assert "test.pdf" in resp.text


# -------------------------------------------------------------------
# Review detail tests
# -------------------------------------------------------------------

def test_review_detail_renders():
    """Detail page shows engine texts for flagged regions."""
    store = {
        "hitl:job123": [
            _sample_region("r1"),
            _sample_region("r2", confidence=0.2),
        ]
    }
    client = _make_test_app(store)
    resp = client.get("/review/job123")
    assert resp.status_code == 200
    assert "text A" in resp.text
    assert "text B" in resp.text
    assert "r1" in resp.text
    assert "r2" in resp.text


def test_review_detail_not_found():
    """Missing job returns 404."""
    client = _make_test_app({})
    resp = client.get("/review/nonexistent")
    assert resp.status_code == 404


# -------------------------------------------------------------------
# Correction submission tests (now with CSRF flow)
# -------------------------------------------------------------------

def test_correction_submission():
    """Submitted corrections update the stored region data (via CSRF flow)."""
    store = {"hitl:job123": [_sample_region("r1")]}
    client = _make_test_app(store)
    resp = _submit_with_csrf(client, "job123", {"r1": "corrected text"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "updated"
    # Verify store was updated
    updated_region = store["hitl:job123"][0]
    assert updated_region["corrected_text"] == "corrected text"


def test_correction_returns_count():
    """Response includes count of corrections applied (via CSRF flow)."""
    store = {
        "hitl:job123": [
            _sample_region("r1"),
            _sample_region("r2"),
        ]
    }
    client = _make_test_app(store)
    resp = _submit_with_csrf(client, "job123", {"r1": "fix A", "r2": "fix B"})
    assert resp.status_code == 200
    assert resp.json()["corrections_applied"] == 2


# -------------------------------------------------------------------
# Token auth tests
# -------------------------------------------------------------------

def test_token_auth_required():
    """GET /review returns 403 when token is set but not provided."""
    store = {"hitl:token": "secret123"}
    client = _make_test_app(store)
    # Without token -- should be 403
    resp = client.get("/review")
    assert resp.status_code == 403
    # With correct token -- should be 200
    resp = client.get("/review?token=secret123")
    assert resp.status_code == 200


def test_token_auth_not_required_when_unset():
    """GET /review returns 200 when no token is configured."""
    store = {}  # no hitl:token key
    client = _make_test_app(store)
    resp = client.get("/review")
    assert resp.status_code == 200


# -------------------------------------------------------------------
# POST token auth tests (AUTH-02) -- updated with CSRF flow
# -------------------------------------------------------------------

def test_post_submit_requires_token():
    """POST /review/{job_id}/submit returns 403 when token is set but not provided."""
    store = {"hitl:token": "secret123", "hitl:job123": [_sample_region("r1")]}
    client = _make_test_app(store)
    resp = client.post("/review/job123/submit", json={"corrections": {"r1": "fix"}})
    assert resp.status_code == 403


def test_post_submit_with_valid_token():
    """POST /review/{job_id}/submit succeeds with correct HITL token and CSRF."""
    store = {"hitl:token": "secret123", "hitl:job123": [_sample_region("r1")]}
    client = _make_test_app(store)
    resp = _submit_with_csrf(client, "job123", {"r1": "fix"}, cookies={"hitl_token": "secret123"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "updated"


def test_post_submit_wrong_token():
    """POST /review/{job_id}/submit returns 403 with wrong token."""
    store = {"hitl:token": "secret123", "hitl:job123": [_sample_region("r1")]}
    client = _make_test_app(store)
    resp = client.post(
        "/review/job123/submit?token=wrong",
        json={"corrections": {"r1": "fix"}},
    )
    assert resp.status_code == 403


# -------------------------------------------------------------------
# Constant-time token comparison tests (AUTH-03)
# -------------------------------------------------------------------

def test_check_token_uses_hmac_compare_digest():
    """router.py must use hmac.compare_digest for constant-time token comparison."""
    import pathlib
    source = pathlib.Path("omniparse/hitl/router.py").read_text()
    assert "import hmac" in source, "hmac module must be imported"
    assert "hmac.compare_digest" in source, "must use hmac.compare_digest"


# -------------------------------------------------------------------
# Cookie auth tests (AUTH-05 -- HITL cookie migration)
# -------------------------------------------------------------------

def test_cookie_auth_accepted():
    """Cookie-based auth should allow access without query param."""
    store = {"hitl:token": "secret123", "hitl:jobs": [_sample_job("job456")]}
    client = _make_test_app(store)
    resp = client.get("/review", cookies={"hitl_token": "secret123"})
    assert resp.status_code == 200
    assert "job456" in resp.text


def test_cookie_set_on_first_query_param_auth():
    """First auth via query param should set hitl_token cookie with secure attributes."""
    store = {"hitl:token": "secret123", "hitl:jobs": [_sample_job("job456")]}
    client = _make_test_app(store)
    resp = client.get("/review?token=secret123")
    assert resp.status_code == 200
    # Check Set-Cookie header
    set_cookie = resp.headers.get("set-cookie", "")
    assert "hitl_token=secret123" in set_cookie
    assert "httponly" in set_cookie.lower()
    assert "secure" in set_cookie.lower()
    assert "samesite=strict" in set_cookie.lower()
    assert "path=/review" in set_cookie.lower()
    assert "max-age=86400" in set_cookie.lower()


def test_cookie_not_reset_when_already_present():
    """Cookie should NOT be re-set when it's already present in the request."""
    store = {"hitl:token": "secret123", "hitl:jobs": [_sample_job("job456")]}
    client = _make_test_app(store)
    resp = client.get(
        "/review?token=secret123",
        cookies={"hitl_token": "secret123"},
    )
    assert resp.status_code == 200
    # The hitl_token cookie should not be re-set (CSRF cookie may be set separately)
    set_cookie = resp.headers.get("set-cookie", "")
    assert "hitl_token=" not in set_cookie


def test_dual_mode_cookie_preferred_over_query():
    """When both cookie and query param are present, cookie takes precedence."""
    store = {"hitl:token": "secret123", "hitl:jobs": [_sample_job("job456")]}
    client = _make_test_app(store)
    # Cookie is correct, query param is wrong -- should succeed (cookie preferred)
    resp = client.get(
        "/review?token=wrong",
        cookies={"hitl_token": "secret123"},
    )
    assert resp.status_code == 200


def test_review_list_links_no_token_param():
    """Review list links should NOT contain ?token= in hrefs."""
    store = {
        "hitl:token": "secret123",
        "hitl:jobs": [_sample_job("job456")],
    }
    client = _make_test_app(store)
    resp = client.get("/review", cookies={"hitl_token": "secret123"})
    assert resp.status_code == 200
    assert "?token=" not in resp.text


def test_review_detail_fetch_has_credentials():
    """Review detail fetch() JS call should include credentials: 'same-origin'."""
    store = {
        "hitl:token": "secret123",
        "hitl:job123": [_sample_region("r1")],
    }
    client = _make_test_app(store)
    resp = client.get("/review/job123", cookies={"hitl_token": "secret123"})
    assert resp.status_code == 200
    assert "credentials" in resp.text


def test_cookie_auth_on_detail_page():
    """Cookie auth should work on detail page."""
    store = {
        "hitl:token": "secret123",
        "hitl:job123": [_sample_region("r1")],
    }
    client = _make_test_app(store)
    resp = client.get("/review/job123", cookies={"hitl_token": "secret123"})
    assert resp.status_code == 200
    assert "r1" in resp.text


def test_cookie_auth_on_post_submit():
    """Cookie auth should work for POST corrections submission (via CSRF flow)."""
    store = {
        "hitl:token": "secret123",
        "hitl:job123": [_sample_region("r1")],
    }
    client = _make_test_app(store)
    resp = _submit_with_csrf(client, "job123", {"r1": "fix"}, cookies={"hitl_token": "secret123"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "updated"


# -------------------------------------------------------------------
# CSRF protection tests (CSRF-01)
# -------------------------------------------------------------------

def test_review_detail_has_csrf_meta_tag():
    """GET /review/{job_id} response HTML contains meta tag with name='csrf-token'."""
    store = {"hitl:job123": [_sample_region("r1")]}
    client = _make_test_app(store)
    resp = client.get("/review/job123")
    assert resp.status_code == 200
    assert '<meta name="csrf-token"' in resp.text


def test_review_detail_sets_csrf_cookie():
    """GET /review/{job_id} response has a Set-Cookie for csrf."""
    store = {"hitl:job123": [_sample_region("r1")]}
    client = _make_test_app(store)
    resp = client.get("/review/job123")
    assert resp.status_code == 200
    set_cookie = resp.headers.get("set-cookie", "")
    assert "fastapi-csrf-token" in set_cookie


def test_submit_without_csrf_rejected():
    """POST /review/{job_id}/submit WITHOUT X-CSRF-Token header returns 403."""
    store = {"hitl:job123": [_sample_region("r1")]}
    client = _make_test_app(store)
    resp = client.post(
        "/review/job123/submit",
        json={"corrections": {"r1": "fix"}},
    )
    assert resp.status_code == 403


def test_submit_with_csrf_succeeds():
    """POST /review/{job_id}/submit WITH valid CSRF token succeeds."""
    store = {"hitl:job123": [_sample_region("r1")]}
    client = _make_test_app(store)
    resp = _submit_with_csrf(client, "job123", {"r1": "fix"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "updated"


def test_review_detail_fetch_has_csrf_header():
    """GET /review/{job_id} HTML source contains 'X-CSRF-Token' in the script block."""
    store = {"hitl:job123": [_sample_region("r1")]}
    client = _make_test_app(store)
    resp = client.get("/review/job123")
    assert resp.status_code == 200
    assert "X-CSRF-Token" in resp.text
