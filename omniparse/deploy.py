"""Deploy entrypoint — imports all modules so Modal discovers decorated classes/functions.

Usage: modal deploy omniparse/deploy.py

This file exists solely to solve Modal's discovery requirement: @app.cls and
@app.function decorators are only registered when their module is imported.
We can't import them in app.py (circular imports), so this file does it.
"""
# Import app first (defines the Modal App and images)
from omniparse.app import app  # noqa: F401

# Import all modules that register @app.cls or @app.function decorators.
# Each import triggers the decorator, registering the class/function with `app`.
# These imports happen locally during `modal deploy`, not inside containers.
import omniparse.pipeline  # noqa: F401 — Pipeline, parse_document, warm_engines
import omniparse.engines.pdfplumber_engine  # noqa: F401 — run_pdfplumber
import omniparse.engines.paddleocr_engine  # noqa: F401 — PaddleOCREngine
import omniparse.engines.docling_engine  # noqa: F401 — DoclingEngine
import omniparse.engines.trocr_engine  # noqa: F401 — TrOCREngine
import omniparse.engines.dots_engine  # noqa: F401 — DotsEngine
import omniparse.llm_arbiter  # noqa: F401 — LLMArbiter
