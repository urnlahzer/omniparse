"""HITL review FastAPI router -- list, view, and correct flagged OCR regions.

Factory function ``create_hitl_router`` accepts a duck-typed ``store`` (dict-like,
compatible with Modal Dict) and returns a configured APIRouter.  Keeping the store
as a parameter (not a global) makes the router Modal-free in tests.

Store key conventions:
- ``hitl:jobs``       -> list[ReviewItem dict]
- ``hitl:{job_id}``   -> list[HitlRegion dict]
- ``hitl:token``      -> optional auth token string
"""
from __future__ import annotations

import hmac
import os
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi_csrf_protect import CsrfProtect
from jinja2 import Environment, FileSystemLoader
from pydantic_settings import BaseSettings

from omniparse.hitl.models import CorrectionPayload, HitlRegion


class CsrfSettings(BaseSettings):
    """Configuration for fastapi-csrf-protect. Loaded via @CsrfProtect.load_config."""

    secret_key: str = os.environ.get("CSRF_SECRET_KEY", "dev-csrf-key-not-for-production")
    cookie_samesite: str = "strict"
    cookie_secure: bool = True
    header_name: str = "X-CSRF-Token"
    header_type: str = ""  # No "Bearer" prefix needed


@CsrfProtect.load_config
def get_csrf_config():
    return CsrfSettings()


def create_hitl_router(
    store: dict,
    *,
    templates_dir: Optional[Path] = None,
) -> APIRouter:
    """Create and return a configured HITL review router.

    Args:
        store: Dict-like object for reading/writing HITL data.
        templates_dir: Path to Jinja2 templates directory.
            Defaults to ``<this_package>/templates/``.

    Returns:
        Configured FastAPI APIRouter with /review endpoints.
    """
    if templates_dir is None:
        templates_dir = Path(__file__).parent / "templates"

    env = Environment(loader=FileSystemLoader(str(templates_dir)), autoescape=True)
    router = APIRouter()

    def _check_token(request: Request) -> str | None:
        """Verify HITL auth via cookie (preferred) or query param (legacy dual-mode).

        Returns verified token on success, None if no auth configured.
        Raises HTTPException(403) if auth configured but token invalid/missing.
        """
        expected = store.get("hitl:token")
        if expected is None:
            return None

        # Prefer cookie (D-05)
        provided = request.cookies.get("hitl_token")
        if provided is None:
            # Legacy fallback: query param (D-06 dual-mode)
            provided = request.query_params.get("token")

        if provided is None or not hmac.compare_digest(provided, expected):
            raise HTTPException(status_code=403, detail="Invalid or missing token")

        return provided

    @router.get("/review", response_class=HTMLResponse)
    async def review_list(request: Request) -> HTMLResponse:
        """List all jobs with HITL-flagged regions."""
        verified_token = _check_token(request)
        try:
            jobs = store["hitl:jobs"]
        except KeyError:
            jobs = []
        template = env.get_template("review_list.html")
        html = template.render(jobs=jobs)
        response = HTMLResponse(content=html)
        # Set cookie on first authenticated visit (migrates from query param to cookie)
        if verified_token and not request.cookies.get("hitl_token"):
            response.set_cookie(
                key="hitl_token",
                value=verified_token,
                httponly=True,
                secure=True,
                samesite="strict",
                path="/review",
                max_age=86400,  # 24 hours per D-07
            )
        return response

    @router.get("/review/{job_id}", response_class=HTMLResponse)
    async def review_detail(
        job_id: str, request: Request, csrf_protect: CsrfProtect = Depends()
    ) -> HTMLResponse:
        """Show all HITL-flagged regions for a specific job."""
        verified_token = _check_token(request)
        try:
            regions_raw = store[f"hitl:{job_id}"]
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        regions = [
            HitlRegion.model_validate(r) if isinstance(r, dict) else r
            for r in regions_raw
        ]
        # Generate CSRF tokens for the correction form
        csrf_token, signed_token = csrf_protect.generate_csrf_tokens()
        template = env.get_template("review_detail.html")
        html = template.render(job_id=job_id, regions=regions, csrf_token=csrf_token)
        response = HTMLResponse(content=html)
        csrf_protect.set_csrf_cookie(signed_token, response)
        if verified_token and not request.cookies.get("hitl_token"):
            response.set_cookie(
                key="hitl_token",
                value=verified_token,
                httponly=True,
                secure=True,
                samesite="strict",
                path="/review",
                max_age=86400,
            )
        return response

    @router.post("/review/{job_id}/submit")
    async def submit_corrections(
        job_id: str, payload: CorrectionPayload, request: Request,
        csrf_protect: CsrfProtect = Depends(),
    ) -> dict:
        """Apply human corrections to flagged regions for a job."""
        _check_token(request)
        await csrf_protect.validate_csrf(request)
        try:
            regions_raw = store[f"hitl:{job_id}"]
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

        regions = [
            HitlRegion.model_validate(r) if isinstance(r, dict) else r
            for r in regions_raw
        ]

        for region in regions:
            if region.region_id in payload.corrections:
                region.corrected_text = payload.corrections[region.region_id]

        # Write back as dicts for store compatibility
        store[f"hitl:{job_id}"] = [r.model_dump() for r in regions]

        return {
            "status": "updated",
            "corrections_applied": len(payload.corrections),
        }

    return router
