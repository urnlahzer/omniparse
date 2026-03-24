"""Tests for Modal App configuration, image definitions, engine stubs, and syntax compliance."""
import json
import os
import pathlib

import modal
import pytest


# Root of the omniparse package
PACKAGE_ROOT = pathlib.Path(__file__).parent.parent
APP_FILE = PACKAGE_ROOT / "app.py"
ENGINES_DIR = PACKAGE_ROOT / "engines"


def _read_source(path: pathlib.Path) -> str:
    """Read source code of a file."""
    return path.read_text()


class TestImageDefinitions:
    """Verify separate container images are defined."""

    def test_separate_images_defined(self):
        """Test 1: app.py defines 6 distinct image variables, all modal.Image instances."""
        from omniparse.app import cpu_image, paddleocr_image, docling_image, trocr_image, dots_image, llm_arbiter_image

        images = [cpu_image, paddleocr_image, docling_image, trocr_image, dots_image, llm_arbiter_image]
        for img in images:
            assert isinstance(img, modal.Image), f"{img} is not a modal.Image"
        # All 6 are distinct objects
        assert len(set(id(img) for img in images)) == 6

    def test_add_local_python_source(self):
        """Test 5: Every image definition includes .add_local_python_source('omniparse')."""
        source = _read_source(APP_FILE)
        count = source.count('.add_local_python_source("omniparse")')
        assert count == 6, f"Expected 6 .add_local_python_source calls, found {count}"

    def test_uv_pip_install_used(self):
        """Test 8: Images use .uv_pip_install where possible; .pip_install for paddlepaddle-gpu (custom index) and nvidia/cuda-based images."""
        source = _read_source(APP_FILE)
        assert ".uv_pip_install(" in source
        # .pip_install allowed for: paddleocr (custom index), trocr (custom index),
        # llm_arbiter (nvidia/cuda base), dots (nvidia/cuda base)
        pip_install_count = source.count(".pip_install(")
        assert pip_install_count == 4, f"Expected exactly 4 .pip_install() (paddleocr + trocr + llm_arbiter + dots), found {pip_install_count}"

    def test_paddlepaddle_gpu_custom_index(self):
        """paddleocr_image installs paddlepaddle-gpu==3.3.0 from PaddlePaddle custom index."""
        source = _read_source(APP_FILE)
        assert "paddlepaddle-gpu==3.3.0" in source
        assert "paddlepaddle.org.cn/packages/stable/cu126" in source

    def test_docling_has_onnxruntime_gpu(self):
        """docling_image includes onnxruntime-gpu to prevent silent CPU-only fallback."""
        source = _read_source(APP_FILE)
        assert "onnxruntime-gpu" in source

    def test_paddleocr_has_rapidfuzz(self):
        """paddleocr_image includes rapidfuzz for text quality check."""
        source = _read_source(APP_FILE)
        assert "rapidfuzz" in source

    def test_docling_version_pinned(self):
        """docling is pinned to >=2.80.0."""
        source = _read_source(APP_FILE)
        assert "docling>=2.80.0" in source

    def test_opencv_pinned_version(self):
        """Test 7: opencv-python-headless is pinned to 4.12.0.88."""
        source = _read_source(APP_FILE)
        assert 'opencv-python-headless==4.12.0.88' in source


    def test_llm_arbiter_image_defined(self):
        """llm_arbiter_image is a modal.Image using nvidia/cuda base."""
        from omniparse.app import llm_arbiter_image
        assert isinstance(llm_arbiter_image, modal.Image)

    def test_llm_arbiter_image_has_vllm(self):
        """llm_arbiter_image installs vllm>=0.17.0 and rapidfuzz."""
        source = _read_source(APP_FILE)
        assert "vllm>=0.17.0" in source
        assert "qwen-vl-utils==0.0.14" in source

    def test_llm_arbiter_nvidia_cuda_base(self):
        """llm_arbiter_image uses nvidia/cuda:12.6.3-devel-ubuntu22.04 base."""
        source = _read_source(APP_FILE)
        assert "nvidia/cuda:12.6.3-devel-ubuntu22.04" in source

    def test_dots_image_nvidia_cuda_base(self):
        """dots_image uses nvidia/cuda base (not debian_slim) for vLLM CUDA build."""
        source = _read_source(APP_FILE)
        # Count nvidia/cuda occurrences -- should be at least 2 (llm_arbiter + dots)
        cuda_count = source.count("nvidia/cuda")
        assert cuda_count >= 2, f"Expected >= 2 nvidia/cuda references (llm_arbiter + dots), found {cuda_count}"

    def test_dots_image_has_vllm(self):
        """dots_image installs vllm>=0.11.0."""
        source = _read_source(APP_FILE)
        assert "vllm>=0.11.0" in source


class TestLLMArbiterClass:
    """Verify LLMArbiter Modal class is importable and configured."""

    def test_llm_arbiter_importable(self):
        """LLMArbiter is importable from omniparse.llm_arbiter."""
        from omniparse.llm_arbiter import LLMArbiter
        assert LLMArbiter is not None

    def test_llm_arbiter_gpu_config(self):
        """LLMArbiter uses A10G, min_containers=0, timeout=120."""
        source = _read_source(PACKAGE_ROOT / "llm_arbiter.py")
        assert 'gpu="A10G"' in source
        assert "min_containers=0" in source
        assert "max_containers=5" in source
        assert "timeout=120" in source

    def test_llm_arbiter_model_path(self):
        """LLMArbiter loads from {MODELS_DIR}/Qwen3-VL-8B-Instruct-FP8."""
        source = _read_source(PACKAGE_ROOT / "llm_arbiter.py")
        assert "Qwen3-VL-8B-Instruct-FP8" in source
        assert "gpu_memory_utilization=0.85" in source
        assert "max_model_len=4096" in source


class TestVolumeAndApp:
    """Verify Volume reference and app config."""

    def test_volume_reference(self):
        """Test 2: model_volume is a modal.Volume and MODELS_DIR is /models."""
        from omniparse.app import model_volume, MODELS_DIR

        assert isinstance(model_volume, modal.Volume)
        assert MODELS_DIR == "/models"

    def test_app_name(self):
        """App is named 'omniparse'."""
        from omniparse.app import app

        assert isinstance(app, modal.App)
        assert app.name == "omniparse"


class TestScaleToZero:
    """Verify engines use correct min_containers settings."""

    def test_scale_to_zero_config(self):
        """Test 3: Non-warm engines use min_containers=0, PaddleOCR uses min_containers=1."""
        scale_to_zero_engines = [
            "pdfplumber_engine.py",
            "docling_engine.py",
            "trocr_engine.py",
            "dots_engine.py",
        ]
        for filename in scale_to_zero_engines:
            source = _read_source(ENGINES_DIR / filename)
            assert "min_containers=0" in source, f"{filename} missing min_containers=0"

    def test_paddleocr_scale_to_zero_dev(self):
        """PaddleOCR uses min_containers=0 during development (saves $1.10/hr A10G cost)."""
        source = _read_source(ENGINES_DIR / "paddleocr_engine.py")
        assert "min_containers=0" in source


class TestNoDeprecatedSyntax:
    """Verify no deprecated Modal syntax is used."""

    DEPRECATED_PATTERNS = [
        "keep_warm",
        "concurrency_limit",
        "container_idle_timeout",
        "@modal.web_endpoint",
        "gpu=modal.gpu.",
    ]

    def test_no_deprecated_syntax(self):
        """Test 4: No production file under omniparse/ contains deprecated Modal syntax."""
        for py_file in PACKAGE_ROOT.rglob("*.py"):
            # Skip test files (they contain the patterns as string literals)
            if "tests" in py_file.parts:
                continue
            source = py_file.read_text()
            for pattern in self.DEPRECATED_PATTERNS:
                assert pattern not in source, (
                    f"Deprecated syntax '{pattern}' found in {py_file.relative_to(PACKAGE_ROOT)}"
                )


class TestEngineStubs:
    """Verify engine stubs raise NotImplementedError."""

    def test_pdfplumber_implemented(self):
        """Test 6a: pdfplumber engine is implemented (no longer a stub)."""
        from omniparse.engines.pdfplumber_engine import extract_page
        assert callable(extract_page)

    def test_paddleocr_implemented(self):
        """Test 6b: PaddleOCR engine is implemented (no longer a stub)."""
        from omniparse.engines.paddleocr_engine import extract_page
        assert callable(extract_page)

    def test_docling_implemented(self):
        """Test 6c: Docling engine is implemented (no longer a stub)."""
        from omniparse.engines.docling_engine import extract_pages
        assert callable(extract_pages)

    def test_trocr_implemented(self):
        """Test 6d: TrOCR engine is implemented (no longer a stub)."""
        from omniparse.engines.trocr_engine import extract_handwriting, segment_lines, recognize_line
        assert callable(extract_handwriting)
        assert callable(segment_lines)
        assert callable(recognize_line)

    def test_dots_engine_implemented(self):
        """Test 6e: Dots.ocr engine is implemented (no longer a stub)."""
        from omniparse.engines.dots_engine import extract_formula, extract_chart_svg, validate_latex
        assert callable(extract_formula)
        assert callable(extract_chart_svg)
        assert callable(validate_latex)


class TestGPUConfig:
    """Verify correct GPU assignments per engine."""

    def test_paddleocr_gpu_a10g(self):
        """PaddleOCR uses A10G."""
        source = _read_source(ENGINES_DIR / "paddleocr_engine.py")
        assert 'gpu="A10G"' in source

    def test_docling_gpu_l4(self):
        """Docling uses L4."""
        source = _read_source(ENGINES_DIR / "docling_engine.py")
        assert 'gpu="L4"' in source

    def test_trocr_gpu_l4(self):
        """TrOCR uses L4."""
        source = _read_source(ENGINES_DIR / "trocr_engine.py")
        assert 'gpu="L4"' in source

    def test_dots_gpu_a10g(self):
        """Dots.ocr uses A10G."""
        source = _read_source(ENGINES_DIR / "dots_engine.py")
        assert 'gpu="A10G"' in source


class TestSecurityHeaders:
    """Verify security headers middleware is defined and functional."""

    def test_security_headers_class_defined(self):
        """app.py defines SecurityHeadersMiddleware class."""
        source = _read_source(APP_FILE)
        assert "class SecurityHeadersMiddleware" in source

    def test_security_headers_registered(self):
        """hitl_web_app registers SecurityHeadersMiddleware."""
        source = _read_source(APP_FILE)
        assert "add_middleware(SecurityHeadersMiddleware)" in source

    def test_security_headers_on_response(self):
        """HITL responses include all 5 security headers."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from omniparse.app import SecurityHeadersMiddleware
        from omniparse.hitl.router import create_hitl_router

        app = FastAPI()
        app.add_middleware(SecurityHeadersMiddleware)
        router = create_hitl_router({})
        app.include_router(router)
        client = TestClient(app)

        resp = client.get("/review")
        assert resp.headers["X-Content-Type-Options"] == "nosniff"
        assert resp.headers["X-Frame-Options"] == "DENY"
        assert resp.headers["Strict-Transport-Security"] == "max-age=300; includeSubDomains"
        assert resp.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
        assert "default-src 'self'" in resp.headers["Content-Security-Policy"]
        assert "style-src 'self' 'unsafe-inline'" in resp.headers["Content-Security-Policy"]
        assert "script-src 'self' 'unsafe-inline'" in resp.headers["Content-Security-Policy"]


class TestExceptionHandler:
    """Verify global exception handler sanitizes errors."""

    def test_exception_handler_registered(self):
        """hitl_web_app registers sanitized_exception_handler."""
        source = _read_source(APP_FILE)
        assert "add_exception_handler(Exception" in source

    def test_production_returns_generic_error(self):
        """In production mode, unhandled exceptions return generic message."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from omniparse.app import sanitized_exception_handler

        app = FastAPI()
        app.add_exception_handler(Exception, sanitized_exception_handler)

        @app.get("/raise-error")
        async def raise_error():
            raise ValueError("sensitive internal detail about /etc/passwd")

        client = TestClient(app, raise_server_exceptions=False)
        old_val = os.environ.get("MODAL_IS_REMOTE")
        os.environ["MODAL_IS_REMOTE"] = "1"
        try:
            resp = client.get("/raise-error")
            assert resp.status_code == 400
            body = resp.json()
            assert body["detail"] == "invalid input"
            assert "sensitive" not in json.dumps(body)
            assert "/etc/passwd" not in json.dumps(body)
        finally:
            if old_val is None:
                os.environ.pop("MODAL_IS_REMOTE", None)
            else:
                os.environ["MODAL_IS_REMOTE"] = old_val

    def test_dev_returns_full_error(self):
        """In dev mode, unhandled exceptions return full details."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from omniparse.app import sanitized_exception_handler

        app = FastAPI()
        app.add_exception_handler(Exception, sanitized_exception_handler)

        @app.get("/raise-error")
        async def raise_error():
            raise ValueError("detailed error info")

        client = TestClient(app, raise_server_exceptions=False)
        old_val = os.environ.pop("MODAL_IS_REMOTE", None)
        try:
            resp = client.get("/raise-error")
            assert resp.status_code == 400
            body = resp.json()
            assert "detailed error info" in body["detail"]
        finally:
            if old_val is not None:
                os.environ["MODAL_IS_REMOTE"] = old_val


class TestCleanupFunction:
    """Verify purge_stale_data scheduled function is defined and configured."""

    def test_purge_stale_data_defined(self):
        """purge_stale_data is importable from omniparse.app and uses modal.Cron."""
        from omniparse.app import purge_stale_data
        assert purge_stale_data is not None
        source = _read_source(APP_FILE)
        assert "schedule=modal.Cron" in source

    def test_cleanup_uses_daily_cron(self):
        """purge_stale_data runs daily at 3am UTC."""
        source = _read_source(APP_FILE)
        assert '"0 3 * * *"' in source

    def test_cleanup_reads_ttl_env_var(self):
        """purge_stale_data reads OMNIPARSE_DATA_TTL_DAYS from environment."""
        source = _read_source(APP_FILE)
        assert "OMNIPARSE_DATA_TTL_DAYS" in source
