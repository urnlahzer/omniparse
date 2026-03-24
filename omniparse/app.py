"""OmniParse Modal App -- central definition for all container images and shared resources."""
import logging
import os

import modal
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest
from starlette.responses import Response

logger = logging.getLogger(__name__)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Inject security headers on every response. Per D-04: custom middleware, not a package."""

    async def dispatch(self, request: StarletteRequest, call_next) -> Response:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        # Per D-06: short max-age (300s) because Modal handles TLS termination
        response.headers["Strict-Transport-Security"] = "max-age=300; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        # Per D-05: permissive enough for HITL Jinja templates with inline styles
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "style-src 'self' 'unsafe-inline'; "
            "script-src 'self' 'unsafe-inline'"
        )
        return response


def _is_production() -> bool:
    """Detect Modal remote environment (production). Per D-03."""
    return os.environ.get("MODAL_IS_REMOTE") == "1"


_ERROR_CATEGORIES: dict[type, tuple[str, int]] = {
    ValueError: ("invalid input", 400),
    KeyError: ("not found", 404),
    FileNotFoundError: ("not found", 404),
    PermissionError: ("forbidden", 403),
}


async def sanitized_exception_handler(request: StarletteRequest, exc: Exception) -> JSONResponse:
    """Return generic error messages in production; full details in dev. Per D-01/D-02/D-03."""
    category, status_code = _ERROR_CATEGORIES.get(type(exc), ("internal error", 500))
    logger.error("Unhandled %s at %s: %s", type(exc).__name__, request.url.path, exc, exc_info=True)

    if _is_production():
        return JSONResponse(status_code=status_code, content={"detail": category})
    else:
        return JSONResponse(
            status_code=status_code,
            content={"detail": str(exc), "type": type(exc).__name__},
        )


app = modal.App("omniparse")

# Shared Volume for model weights (~13 GB across all engines)
# Phase 1: mostly empty (pdfplumber needs no model weights)
# Phase 2+: PaddleOCR, Docling, TrOCR, Dots.ocr, Qwen3-VL weights
model_volume = modal.Volume.from_name("ocr-models", create_if_missing=True)

MODELS_DIR = "/models"

# --- Container Images (one per engine, mandatory separation) ---

# CPU image for preprocessing + pdfplumber (Phase 1: working)
cpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("poppler-utils")
    .uv_pip_install(
        "pdfplumber==0.11.9",
        "Pillow",
        "opencv-python-headless==4.12.0.88",
        "pdf2image",
        "numpy",
        "pydantic>=2.0",
        "fastapi[standard]",
        "sequence-align",
        "rapidfuzz>=3.14",
        "scipy>=1.14",
        "slowapi>=0.1.9",
        "fastapi-csrf-protect>=1.0.7",
        "pydantic-settings>=2.0",
    )
    .add_local_python_source("omniparse")
    .add_local_dir("omniparse/hitl/templates", remote_path="/root/templates")
)

# GPU image for PaddleOCR PP-StructureV3 (Phase 2)
# NOTE: paddlepaddle-gpu MUST come from PaddlePaddle's custom index (PyPI only has 2.6.2)
# Models baked into image to avoid ~60s HuggingFace downloads on cold start.
# Bump _PADDLE_MODEL_REV to force re-download on next deploy.
_PADDLE_MODEL_REV = "2026-03-21"  # bump to refresh cached models
paddleocr_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgomp1", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "paddlepaddle-gpu==3.3.0",
        extra_index_url="https://www.paddlepaddle.org.cn/packages/stable/cu126/",
    )
    .uv_pip_install(
        "paddleocr>=3.4.0",
        "numpy",
        "pydantic>=2.0",
        "rapidfuzz>=3.14",
    )
    # paddlex[ocr] is required for PP-StructureV3 pipeline
    # Force headless opencv AFTER to avoid libGL dependency
    .run_commands(
        'pip install "paddlex[ocr]"',
        "pip install --force-reinstall opencv-python-headless==4.12.0.88",
    )
    .env({"PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True"})
    # Pre-download all PP-StructureV3 + PP-OCRv5 sub-models into the image.
    # Uses huggingface_hub Python API (no paddle import, avoids libcuda.so at build time).
    # PaddleX checks /root/.paddlex/official_models/<name>/ for cached models.
    .run_commands(
        f"echo 'paddle_model_rev={_PADDLE_MODEL_REV}' && "
        "python -c \""
        "from huggingface_hub import snapshot_download; "
        "models = ["
        "'PP-LCNet_x1_0_doc_ori','UVDoc','PP-LCNet_x1_0_textline_ori',"
        "'PP-OCRv5_server_det','PP-OCRv5_server_rec','en_PP-OCRv5_mobile_rec',"
        "'PP-DocBlockLayout','PP-DocLayout_plus-L','PP-LCNet_x1_0_table_cls',"
        "'SLANet_plus','SLANeXt_wired',"
        "'RT-DETR-L_wired_table_cell_det','RT-DETR-L_wireless_table_cell_det',"
        "'PP-FormulaNet_plus-L','PP-Chart2Table']; "
        "[snapshot_download(f'PaddlePaddle/{m}', local_dir=f'/root/.paddlex/official_models/{m}') for m in models]; "
        "print(f'Downloaded {len(models)} PaddleOCR models')"
        "\"",
    )
    .add_local_python_source("omniparse")
)

# GPU image for Docling (Phase 2)
# NOTE: onnxruntime-gpu required to prevent silent CPU-only fallback (GitHub #2528)
# Models baked into image to avoid ~30s downloads on cold start.
# Bump _DOCLING_MODEL_REV to force re-download on next deploy.
_DOCLING_MODEL_REV = "2026-03-21"  # bump to refresh cached models
docling_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .uv_pip_install(
        "docling>=2.80.0",
        "onnxruntime-gpu",
        "torch",
        "torchvision",
        "opencv-python-headless==4.12.0.88",
        "numpy",
        "pydantic>=2.0",
        extra_index_url="https://download.pytorch.org/whl/cu128",
        extra_options="--index-strategy unsafe-best-match",
    )
    # Pre-download Docling's layout + OCR models into the image.
    # DocumentConverter() triggers download of all pipeline models.
    .run_commands(
        f"echo 'model_rev={_DOCLING_MODEL_REV}' && python -c \""
        "from docling.document_converter import DocumentConverter; "
        "DocumentConverter()"
        "\"",
    )
    .add_local_python_source("omniparse")
)

# GPU image for TrOCR handwriting (Phase 4)
# NOTE: Contains both PaddlePaddle-GPU (DBNet line segmentation) and
# PyTorch (TrOCR inference). Per CONTEXT.md decision: DBNet runs inside
# TrOCR container. Both frameworks pinned to CUDA 12.6 for compatibility.
trocr_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgomp1")
    .pip_install(
        "paddlepaddle-gpu==3.3.0",
        extra_index_url="https://www.paddlepaddle.org.cn/packages/stable/cu126/",
    )
    .uv_pip_install(
        "transformers>=4.40",
        "torch",
        "torchvision",
        "paddleocr>=3.4.0",
        "opencv-python-headless==4.12.0.88",
        "Pillow",
        "pydantic>=2.0",
    )
    .add_local_python_source("omniparse")
)

# GPU image for Dots.ocr formulas/charts (Phase 4)
# Uses nvidia/cuda base because vLLM needs CUDA toolkit at build time
# (same pattern as llm_arbiter_image). Dots.ocr-1.5 served via vLLM >=0.11.0.
dots_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.3-devel-ubuntu22.04",
        add_python="3.11",
    )
    .pip_install(
        "vllm>=0.11.0",
        "Pillow",
        "pydantic>=2.0",
    )
    .add_local_python_source("omniparse")
)

# GPU image for LLM arbiter -- Qwen3-VL-8B-Instruct-FP8 via vLLM (Phase 3)
# Own container on A10G, separate from all engines. Per CONTEXT.md: independent scaling, no memory contention.
# Uses nvidia/cuda base image because vLLM needs CUDA toolkit at build time.
llm_arbiter_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.3-devel-ubuntu22.04",
        add_python="3.11",
    )
    .pip_install(
        "vllm>=0.17.0",
        "qwen-vl-utils==0.0.14",
        "Pillow",
        "pydantic>=2.0",
        "rapidfuzz>=3.14",
    )
    .add_local_python_source("omniparse")
)


# --- HITL Review Web App (Phase 6) ---

@app.function(image=cpu_image)
@modal.asgi_app()
def hitl_web_app():
    """HITL review interface served as ASGI web app.

    Connects to Modal Dict for review data storage and serves
    Jinja2 templates for human review of flagged OCR regions.
    """
    from pathlib import Path

    from fastapi import FastAPI

    from omniparse.hitl.router import create_hitl_router

    from slowapi.errors import RateLimitExceeded
    from slowapi.middleware import SlowAPIMiddleware

    from omniparse.api.rate_limit import limiter, rate_limit_exceeded_handler

    hitl_store = modal.Dict.from_name("omniparse-hitl", create_if_missing=True)
    web = FastAPI(title="OmniParse HITL Review")
    web.state.limiter = limiter
    web.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)
    web.add_middleware(SlowAPIMiddleware)
    web.add_middleware(SecurityHeadersMiddleware)
    web.add_exception_handler(Exception, sanitized_exception_handler)

    # In Modal, templates are mounted at /root/templates
    # Locally, fall back to package-relative path
    modal_templates = Path("/root/templates")
    templates_dir = modal_templates if modal_templates.exists() else None

    router = create_hitl_router(hitl_store, templates_dir=templates_dir)
    web.include_router(router)
    return web


# --- Regression Test Suite (Phase 6) ---

@app.function(
    image=cpu_image,
    schedule=modal.Cron("0 6 * * 1"),  # Every Monday at 6am UTC
    timeout=1800,  # 30 minutes for full benchmark suite
)
def run_regression_suite():
    """Process benchmark documents and compare against stored baselines.

    Runs weekly on Monday 6am UTC. Loads benchmark docs from Volume,
    processes through pipeline, computes CER metrics, and compares
    against baselines. Alerts on regressions.
    """
    import logging as _logging

    from omniparse.regression.runner import run_benchmark, check_baselines, format_regression_report
    from omniparse.regression.baselines import BASELINES

    _logger = _logging.getLogger(__name__)

    # Load benchmarks from Volume (ground truth stored alongside docs)
    benchmarks_vol = modal.Volume.from_name("omniparse-benchmarks", create_if_missing=True)

    pipeline_cls = modal.Cls.from_name("omniparse", "Pipeline")
    pipeline = pipeline_cls()

    results = {}
    for doc_name in BASELINES:
        try:
            doc_path = f"/benchmarks/{doc_name}"
            gt_path = f"/benchmarks/{doc_name}.gt.txt"
            import os
            if not os.path.exists(doc_path) or not os.path.exists(gt_path):
                _logger.warning("Benchmark %s: files not found at %s, skipping", doc_name, doc_path)
                continue
            with open(doc_path, "rb") as f:
                doc_bytes = f.read()
            with open(gt_path, "r") as f:
                gt_text = f.read()
            _logger.info("Processing benchmark: %s", doc_name)
            result = run_benchmark(doc_name, doc_bytes, gt_text, pipeline.process.remote)
            results[doc_name] = result
        except Exception as e:
            _logger.error("Benchmark %s failed: %s", doc_name, e)

    if results:
        regressions = check_baselines(results, BASELINES)
        report = format_regression_report(regressions)
        _logger.info(report)

        if regressions:
            _logger.warning("REGRESSION DETECTED: %d metrics below baseline", len(regressions))


# --- Data Retention Cleanup (DATA-01) ---

@app.function(
    image=cpu_image,
    schedule=modal.Cron("0 3 * * *"),  # Daily at 3am UTC
    timeout=600,  # 10 minutes should be plenty for cleanup
)
def purge_stale_data():
    """Purge expired entries from Modal Dicts per DATA-01.

    Runs daily. Deletes job metadata, HITL review data, and spend
    bucket keys older than OMNIPARSE_DATA_TTL_DAYS (default 30).
    """
    import logging as _logging
    import os

    from omniparse.api.cleanup import (
        DEFAULT_TTL_DAYS,
        purge_expired_entries,
        purge_expired_spend_buckets,
    )

    _logger = _logging.getLogger(__name__)

    ttl_days = int(os.environ.get("OMNIPARSE_DATA_TTL_DAYS", str(DEFAULT_TTL_DAYS)))

    jobs_store = modal.Dict.from_name("omniparse-jobs", create_if_missing=True)
    hitl_store = modal.Dict.from_name("omniparse-hitl", create_if_missing=True)

    jobs_purged = purge_expired_entries(jobs_store, ttl_days=ttl_days)
    hitl_purged = purge_expired_entries(hitl_store, ttl_days=ttl_days)

    # Spend buckets live in the same store as API keys
    api_keys_store = modal.Dict.from_name("omniparse-api-keys", create_if_missing=True)
    buckets_purged = purge_expired_spend_buckets(api_keys_store, ttl_days=ttl_days)

    _logger.info(
        "TTL cleanup complete: %d jobs, %d HITL entries, %d spend buckets purged (TTL=%d days)",
        jobs_purged, hitl_purged, buckets_purged, ttl_days,
    )
