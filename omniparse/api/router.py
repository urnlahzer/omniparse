"""Async document submission API: submit, status, result endpoints.

Uses Modal FunctionCall for async job management and Modal Dict for
job metadata storage. All auth/webhook/cost functions imported from
sibling modules (pure functions, tested independently).
"""
import hashlib
import hmac
import logging
import time
from typing import Callable, Optional

from fastapi import APIRouter, Form, Header, HTTPException, UploadFile
from starlette.requests import Request

from omniparse.api.auth import validate_api_key
from omniparse.api.models import SubmitResponse, JobStatus
from omniparse.api.rate_limit import limiter
from omniparse.api.validation import validate_callback_url, validate_upload, MAX_UPLOAD_BYTES

logger = logging.getLogger(__name__)


def create_api_router(
    api_keys_store: dict,
    jobs_store: dict,
    webhook_secret: str,
    spawn_fn: Optional[Callable] = None,
) -> APIRouter:
    """Create and return a configured API router for async document submission.

    Args:
        api_keys_store: Dict-like store for API key validation
            (Modal Dict in production, plain dict in tests).
        jobs_store: Dict-like store for job metadata
            (Modal Dict in production, plain dict in tests).
        webhook_secret: Shared secret for webhook HMAC-SHA256 signing.
        spawn_fn: Callable ``(file_bytes, filename, budget_usd, **kwargs) -> obj``
            returning an object with ``.object_id`` attribute.  When None,
            defaults to spawning via ``modal.Cls.from_name("omniparse", "Pipeline")``.

    Returns:
        Configured FastAPI APIRouter with /submit, /status, /result endpoints.
    """
    if spawn_fn is None:  # pragma: no cover -- Modal-only path
        def _default_spawn(
            file_bytes: bytes,
            filename: str,
            budget_usd: float | None,
            **kwargs,
        ):
            import modal
            pipeline = modal.Cls.from_name("omniparse", "Pipeline")()
            return pipeline.process.spawn(
                file_bytes, filename, budget_usd=budget_usd, **kwargs,
            )
        spawn_fn = _default_spawn

    router = APIRouter()

    def _verify_ownership(job_id: str, x_api_key: str) -> dict:
        """Fetch job metadata and verify the requesting key owns it.

        Returns job metadata dict on success.
        Raises HTTPException(404) for unknown job, HTTPException(403) for wrong key.
        """
        try:
            meta = jobs_store[job_id]
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        request_hash = hashlib.sha256(x_api_key.encode()).hexdigest()
        stored_hash = meta.get("api_key_hash", "")
        if not hmac.compare_digest(request_hash, stored_hash):
            raise HTTPException(status_code=403, detail="Access denied")
        return meta

    @router.post("/submit", response_model=SubmitResponse)
    @limiter.limit("5/minute")
    async def submit_document(
        request: Request,
        file: UploadFile,
        callback_url: str | None = Form(default=None),
        ce_threshold: float = Form(default=0.4),
        confidence_floor: float = Form(default=0.0),
        x_api_key: str = Header(...),
    ) -> SubmitResponse:
        """Submit a document for async processing.

        Validates the API key, callback URL (SSRF), file size/format,
        then spawns a Pipeline.process call and optionally schedules a
        webhook notification.
        """
        # 1. Validate API key
        try:
            entry = validate_api_key(api_keys_store, x_api_key)
        except ValueError as exc:
            logger.error("API key validation failed: %s", exc)
            if "Budget exhausted" in str(exc):
                raise HTTPException(status_code=429, detail="Budget exhausted")
            raise HTTPException(status_code=401, detail="Invalid API key")

        # 2. Extract budget_usd from validated entry
        budget_usd = entry["budget_usd"]

        # 2a. SSRF validation -- fail early before file read (D-05)
        if callback_url:
            try:
                validate_callback_url(callback_url)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc))

        # 2b. Content-Length pre-check -- reject before reading body (D-06)
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="File too large")

        # 3. Read file bytes
        file_bytes = await file.read()
        filename = file.filename or "document.pdf"

        # 3a. File validation -- size + PDF magic (D-06, D-07, D-08)
        try:
            validate_upload(file_bytes, filename)
        except ValueError as exc:
            raise HTTPException(
                status_code=413 if "limit" in str(exc) else 422,
                detail=str(exc),
            )

        # 4. Spawn pipeline async with budget_usd and consensus tuning params
        call = spawn_fn(
            file_bytes, filename, budget_usd,
            ce_threshold=ce_threshold,
            confidence_floor=confidence_floor,
        )

        # 5. Store job metadata (hash API key -- never store plaintext)
        jobs_store[call.object_id] = {
            "api_key_hash": hashlib.sha256(x_api_key.encode()).hexdigest(),
            "callback_url": callback_url,
            "ce_threshold": ce_threshold,
            "confidence_floor": confidence_floor,
            "budget_usd": budget_usd,
            "status": "processing",
            "filename": filename,
            "created_at": time.time(),  # TTL cleanup (DATA-01, D-08)
        }

        # 6. Schedule webhook notification if callback_url provided
        if callback_url:
            try:
                from omniparse.pipeline import notify_on_complete
                notify_on_complete.spawn(call.object_id, callback_url)
            except Exception:
                pass  # Best effort -- webhook notifier is non-critical

        # 7. Return response
        return SubmitResponse(job_id=call.object_id, status="processing")

    @router.get("/status/{job_id}", response_model=JobStatus)
    @limiter.limit("30/minute")
    async def get_status(
        request: Request,
        job_id: str,
        x_api_key: str | None = Header(default=None),
    ) -> JobStatus:
        """Poll current status of a submitted job. Requires API key ownership."""
        if x_api_key is None:
            raise HTTPException(status_code=401, detail="Missing API key")
        try:
            validate_api_key(api_keys_store, x_api_key)
        except ValueError:
            raise HTTPException(status_code=401, detail="Invalid API key")
        meta = _verify_ownership(job_id, x_api_key)
        return JobStatus(
            job_id=job_id,
            status=meta["status"],
            result=meta.get("result"),
            error=meta.get("error"),
        )

    @router.get("/result/{job_id}")
    @limiter.limit("30/minute")
    async def get_result(
        request: Request,
        job_id: str,
        x_api_key: str | None = Header(default=None),
    ) -> dict:
        """Retrieve the result of a completed job. Requires API key ownership."""
        if x_api_key is None:
            raise HTTPException(status_code=401, detail="Missing API key")
        try:
            validate_api_key(api_keys_store, x_api_key)
        except ValueError:
            raise HTTPException(status_code=401, detail="Invalid API key")
        meta = _verify_ownership(job_id, x_api_key)

        if meta["status"] != "completed":
            # Return 202 Accepted with current status
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=202,
                content={"status": meta["status"]},
            )

        return meta["result"]

    return router
