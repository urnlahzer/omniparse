<!-- GSD:project-start source:PROJECT.md -->
## Project

**OmniParse Security Hardening**

Security hardening of OmniParse, a multi-engine OCR ensemble pipeline running on Modal. A comprehensive security review identified 19 findings (4 high, 9 medium, 4 low) across authentication, authorization, input validation, secrets management, and infrastructure. This project fixes all 19 findings in priority order.

**Core Value:** Every endpoint that handles user data must enforce authentication and reject malicious input — unauthenticated access to OCR results containing PII/PHI is the highest-risk gap.

### Constraints

- **Dependencies**: Fewer is better. Prefer stdlib solutions (e.g., `hmac.compare_digest`) over new packages. Only add a dependency when there's no reasonable alternative.
- **Modal cost**: Minimize anything that increases cold start time or runtime. Lightweight middleware only.
- **API stability**: Existing API clients depend on current endpoints. Auth additions should be backward-compatible where possible (e.g., require header but don't change response shape). Document any breaking changes.
- **Priority order**: P1 (high) fixes first, then P2 (medium), then P3 (low). This is the security review's priority classification.
<!-- GSD:project-end -->

<!-- GSD:stack-start source:codebase/STACK.md -->
## Technology Stack

## Languages
- Python 3.11+ - All orchestration, engine integration, API endpoints, and ML inference
## Runtime
- Python 3.11 (specified in `pyproject.toml` as `requires-python = ">=3.11"`)
- Modal serverless runtime (GPU and CPU containers on demand)
- pip (uv_pip_install in Modal for faster dependency resolution)
- Lockfile: pyproject.toml (PEP 517/518 format, no lock file generated)
## Frameworks
- FastAPI 0.100+ - HTTP API server for async document submission and webhook handling
- Modal 0.x - Serverless GPU orchestration, function/class decoration, volume/dict storage, scheduled jobs
- Jinja2 3.1+ - Template rendering for HITL review web interface (`omniparse/hitl/templates`)
- pytest 8.0+ - Test runner and framework
- uv - Ultra-fast pip resolver (used in Modal image builds)
## Key Dependencies
- pydantic>=2.0 - Data validation for API models, region contracts, pipeline outputs
- pdfplumber>=0.10 - Text extraction from PDFs with character-level bounding boxes (CPU engine, always-run)
- Pillow>=10.0 - Image processing, PDF-to-image conversion, cropping for LLM input
- numpy>=1.24 - Numerical operations (IoU calculation, alignment, coordinate transforms)
- opencv-python-headless>=4.8 - Image preprocessing (crop, resize, normalize)
- pdf2image>=1.16 - PDF to PIL Image conversion for multi-engine input
- httpx>=0.27 - Async HTTP client for webhook delivery with HMAC signing
- paddlepaddle-gpu==3.3.0 - GPU framework for PaddleOCR PP-StructureV3 (layout + OCR)
- paddleocr>=3.4.0 - Baidu's OCR engine with table structure detection
- paddlex[ocr] - PaddleX framework for PP-StructureV3 pipeline and models
- docling>=2.80.0 - AirBnB's document converter with hierarchical layout detection
- torch - PyTorch for Docling and TrOCR inference
- torchvision - Vision models for PyTorch-based engines
- transformers>=4.40 - Hugging Face transformers for TrOCR and Qwen3-VL
- vllm>=0.11.0 - LLM serving (Dots.ocr formula/chart recognition)
- vllm>=0.17.0 - LLM serving (Qwen3-VL-8B arbiter)
- qwen-vl-utils==0.0.14 - Qwen3-VL image encoding utilities
- sequence-align>=0.3 - Needleman-Wunsch text alignment for cross-engine matching
- rapidfuzz>=3.14 - Fast string similarity and edit distance (alignment, arbiter hallucination guards)
- scipy>=1.14 - Scientific computing (IOU calculations, consensus voting)
- fpdf2>=2.7 - Synthetic PDF generation for regression test suite
- PyYAML>=6.0 - Baseline config parsing
- huggingface-hub>=0.24 - Model download from Hugging Face (PaddleOCR, Docling, Qwen3-VL pre-cached in images)
- onnxruntime-gpu - GPU acceleration for Docling (prevents silent CPU fallback)
- CUDA 12.6/12.8 - GPU compute (baked into container images via nvidia/cuda base)
## Configuration
- `OMNIPARSE_SAMPLES_DIR` - Path to PDF sample directory for integration tests
- `OMNIPARSE_WEBHOOK_SECRET` - HMAC-SHA256 shared secret for webhook signing
- `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True` - Disable PaddleX model source validation (allows local cache)
- `pyproject.toml` - Package metadata, dependencies, test configuration
- Modal image definitions in `omniparse/app.py`:
## Platform Requirements
- Python 3.11+
- Modal CLI (authentication via `modal token new`)
- ~5GB local disk (for .venv and test cache)
- Modal account with GPU quota allocation:
- Modal Volume for model weights (~19GB total across all engines)
- Modal Dict for job metadata storage and HITL review data
- Modal Volume for benchmark suite (regression testing)
- `modal deploy omniparse/deploy.py` - Builds and deploys all containers
- First deploy: 10-15 minutes (downloads 5GB+ dependencies)
- Subsequent deploys: <5 seconds (images cached)
- Model weight download: 5-10 minutes on first `modal run omniparse/setup_volume.py`
<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->
## Conventions

## Naming Patterns
- Snake case with underscores: `consensus.py`, `markdown_compiler.py`, `paddleocr_engine.py`
- Test files prefixed with `test_`: `test_consensus.py`, `test_pipeline.py`
- Feature-specific modules in subdirectories: `engines/`, `api/`, `models/`, `regression/`, `hitl/`
- Snake case: `compute_region_ce()`, `weighted_majority_vote()`, `preprocess()`, `normalize_bbox()`
- Private functions prefixed with underscore: `_default_engines()`, `_is_noise()`, `_make_image_with_text_lines()`
- Async functions use `async def`: `async def submit_document()`
- Prefixes indicate function purpose: `compute_*` (calculations), `filter_*` (filtering), `validate_*` (validation)
- Snake case throughout: `bbox_px`, `page_height`, `consensus_text`, `engine_texts`
- Module-level constants in UPPER_CASE: `CE_THRESHOLD = 0.4`, `TARGET_DPI = 300`, `ACCEPTED_EXTENSIONS`, `VALID_ELEMENT_TYPES`
- Constants include explanatory comments at module level
- Dictionary keys use snake_case: `{"pdfplumber": 1.0, "paddleocr": 1.0}`
- Pydantic models for data contracts: `Region`, `EngineOutput`, `PagePayload`, `AlignedRegion`, `ConsensusResult`
- Type hints everywhere (Python 3.11+): `def compute_iou(bbox_a: list[float], bbox_b: list[float]) -> float:`
- Optional types: `page_height: float | None = None` (Python 3.10+ union syntax)
- Generic type hints: `dict[str, list[str]]`, `list[Region]`, `Optional[dict]`
## Code Style
- No explicit formatter configured (black/ruff/autopep8 not required)
- Lines appear to follow natural Python conventions with reasonable length (80-100 characters typical)
- Indentation: 4 spaces (Python standard)
- No `.pylintrc`, `.flake8`, or `ruff.toml` configured
- Code appears to follow PEP 8 conventions naturally
- Import style is consistent across codebase
## Import Organization
- No configured path aliases (no tsconfig or import hooks)
- All imports use full relative paths: `from omniparse.models.region import Region`
- Barrel files exist in `__init__.py` for re-exporting: `omniparse/models/__init__.py` re-exports `Region`, `EngineOutput`, `PagePayload`
- Used strategically to avoid circular dependencies: `from omniparse.app import app, cpu_image` inside `process_document()` in `pipeline.py`
- Also used for optional dependencies: `try: from fastapi import UploadFile except ImportError`
- Imported inside test methods when creating fixtures to avoid dependencies: `from fpdf import FPDF` inside test functions
## Error Handling
- Raise `ValueError` for validation failures or invalid input: `raise ValueError(f"Unknown coordinate system: {source_system!r}...")`
- Raise `HTTPException` in API endpoints for HTTP-specific errors: `raise HTTPException(status_code=401, detail=msg)`
- Use descriptive error messages with context: `raise ValueError("page_height required for docling_bottomleft conversion")`
- No bare `except` clauses — catch specific exceptions: `except ValueError as exc:`, `except KeyError:`, `except ValidationError:`
- Pydantic `ValidationError` caught for model validation failures in tests and API endpoints
- Custom exceptions wrapped in try/except blocks in consensus and API modules
- Minimal use of try/except — functions prefer explicit validation upfront
- Where used, exceptions are caught specifically and re-raised as domain-specific errors
- Example from `api/auth.py`: catches `KeyError` and raises `ValueError` with message
## Logging
- Module-level logger: `logger = logging.getLogger(__name__)` in each module
- Used for informational messages and warnings: `logger.info()`, `logger.warning()`, `logger.error()`
- Logging in production-critical paths: pipeline stages, engine invocations, consensus decisions
- Example from `consensus.py`: `logger = logging.getLogger(__name__)` at module top
- Engine execution start/completion
- Consensus decision pathways (voting vs arbitration)
- Major preprocessing steps
- Cost tracking and budget checks
- Errors and exceptions
## Comments
- At function docstrings: All public functions and classes have docstrings
- For non-obvious algorithms: NW alignment parameters, CE threshold rationale
- For known limitations: ALGN-04 notes visually-similar OCR pair scoring deferred to v2
- For integration points: Comments explaining why local imports are used
- Design decisions: Constants have explanatory comments (e.g., `CE_THRESHOLD = 0.4 # chosen as starting point per research`)
- Using Google-style docstrings (not Google docstring syntax, but similar structure)
- Format: Module docstring, function docstring with Args/Returns/Raises sections
- Example from `normalization.py`:
- Every module starts with a docstring explaining purpose, design pattern, and entry points
- Example from `consensus.py`:
## Function Design
- Typically 20-50 lines for algorithmic functions
- Pure functions preferred (no side effects)
- Longer functions (100+ lines) have clear section breaks with comment blocks:
- Use keyword-only arguments for clarity: `def process_document(..., budget_usd: float | None = None):`
- Prefix keyword-only args with `*`: `normalize_bbox(bbox, source_system, dpi=300, *, page_height=None)`
- Type hints required on all parameters
- Default values for optional parameters: `dpi: int = 300`, `format: str = "PNG"`
- Single return value (tuple for multiple returns when multiple distinct things returned)
- Type hints required: `-> list[PagePayload]`, `-> float`, `-> dict[str, list[str]]`
- Functions that compute statistics return floats: `-> float`
- Data processing functions return model instances: `-> list[Region]`, `-> ConsensusResult`
## Module Design
- Public API exported from `__init__.py` files: `omniparse/models/__init__.py` exports key models
- Modules use `from x import y` explicitly; no `import *`
- Each module is self-contained and imports only what it needs
- `omniparse/models/__init__.py` re-exports: `Region`, `EngineOutput`, `PagePayload`
- `omniparse/engines/__init__.py` exists but minimal (future proofing)
- Consumers can `from omniparse.models import Region` instead of deep imports
- Each module has single responsibility: `consensus.py` (decision logic), `alignment.py` (spatial matching), `normalization.py` (coordinate conversion)
- Shared data models in `models/` subdirectory
- Engine-specific logic isolated in `engines/` subdirectory
- API-specific logic in `api/` subdirectory
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->
## Architecture

## Pattern Overview
- Concurrent engine execution (3 always-run engines: pdfplumber, PaddleOCR, Docling)
- Layout-aware specialist dispatch (handwriting → TrOCR, formulas → Dots.ocr, charts → Dots.ocr)
- Progressive consensus resolution (alignment → voting → LLM arbitration → HITL escalation)
- Pure function isolation (all logic except Modal invocation is testable without GPU)
- Comprehensive processing audit trail (per-region decision logging for compliance)
## Layers
- Purpose: Normalize input documents (PDF, PNG, JPG, TIFF) to consistent format
- Location: `omniparse/preprocess.py`
- Contains: DPI normalization, skew correction, landscape rotation, page splitting
- Depends on: pdf2image, Pillow, OpenCV
- Used by: Pipeline orchestrator (entry point)
- Output: List of PagePayload objects (one per page with normalized image bytes)
- Purpose: Run OCR/document analysis engines in parallel across pages
- Location: `omniparse/engines/` (pdfplumber_engine.py, paddleocr_engine.py, docling_engine.py, trocr_engine.py, dots_engine.py)
- Contains: Modal-wrapped engine classes with `.run()` or `.run.map()` methods
- Depends on: Modal, engine-specific libraries (pdfplumber, PaddleOCR, Docling, TrOCR, vLLM)
- Used by: Pipeline.process_document()
- Output: EngineOutput objects containing list of Region objects per page
- Purpose: Clean and normalize raw engine outputs before consensus
- Location: `omniparse/noise_filter.py`, `omniparse/normalization.py`, `omniparse/docling_premerge.py`
- Contains: Noise removal (line numbers, page numbers, footers), coordinate normalization [0,1], Docling word-level fragment merging
- Depends on: Models (Region, EngineOutput)
- Used by: Pipeline per-page loop
- Output: Cleaned EngineOutput objects
- Purpose: Route layout-classified regions to specialized engines
- Location: `omniparse/dispatch.py`
- Contains: Element type classification (handwriting, formula, chart) based on PaddleOCR output
- Depends on: PaddleOCR output with element_type field
- Used by: Pipeline per-page loop (after PaddleOCR available)
- Output: Dict mapping engine names to Region lists for dispatch
- Purpose: Match regions across engines by spatial overlap (IoU) and element type
- Location: `omniparse/alignment.py`
- Contains: IoU computation, hierarchical clustering for region grouping, Needleman-Wunsch text alignment
- Depends on: scipy, sequence-align library
- Used by: Consensus module
- Output: AlignedRegion objects with per-engine text variants and confidence scores
- Purpose: Resolve text disagreements through voting, entropy thresholding, and LLM arbitration
- Location: `omniparse/consensus.py`, `omniparse/llm_arbiter.py`
- Contains: Character Entropy (CE) computation, weighted majority voting, LLM orchestration
- Depends on: Alignment output, LLM arbiter Modal class
- Used by: Pipeline per-page loop
- Output: ConsensusResult with final text and resolution metadata (voting/arbitration/hitl_fallback)
- Purpose: Transform consensus output into GFM Markdown with YAML frontmatter
- Location: `omniparse/markdown_compiler.py`
- Contains: Header hierarchy preservation, table formatting, formula/chart rendering, HITL flags
- Depends on: ConsensusResult, yaml library
- Used by: Pipeline final step
- Output: Single Markdown string
- Purpose: Track processing metrics, costs, and region-level decisions for audit trails
- Location: `omniparse/observability.py`
- Contains: ProcessingLogBuilder class accumulating per-page/per-region data, cost estimation
- Depends on: Modal pricing constants, ProcessingLog models
- Used by: Pipeline throughout execution
- Output: ProcessingLog with per-region audit trails
- Purpose: HTTP endpoint exposure and async job management
- Location: `omniparse/api/router.py`, `omniparse/api/auth.py`, `omniparse/api/cost_guard.py`, `omniparse/api/webhooks.py`
- Contains: /submit, /status, /result endpoints; API key validation; budget enforcement; webhook callbacks
- Depends on: FastAPI, Modal Dict for job storage
- Used by: HTTP clients
- Output: SubmitResponse, JobStatus, final PipelineResult
## Data Flow
- Preprocess fails → PipelineResult with empty markdown, log records error, metadata.page_count=0
- Engine timeout → Engine output skipped, region processed with fewer engines
- Budget exceeded → Cost guard raises BudgetExceededError at submission time
- LLM rejection → Voting fallback used, source="hitl_fallback", hitl_flag=True in RegionLog
- Webhook dispatch fails → Non-critical, logged but doesn't block job completion
- Modal Dict stores job metadata (status, result, error, budget_usd, callback_url)
- ProcessingLogBuilder accumulates state during pipeline execution
- No shared mutable state between pages (thread-safe)
- Engine results materialized as lists before per-page loop
## Key Abstractions
- Purpose: Canonical data contract for OCR extraction (location + text + metadata)
- Examples: `omniparse/models/region.py`
- Pattern: Pydantic BaseModel with mandatory fields (id, element_type, bounding_box, confidence, text_content) and optional metadata (table_structure, coordinates_norm)
- Used by: All engines, alignment, consensus modules
- Purpose: Wrapper for per-engine, per-page results
- Examples: `omniparse/models/region.py`
- Pattern: Pydantic BaseModel with page number, engine name, and list of Region objects
- Serialized as JSON across Modal remote boundaries
- Purpose: Intermediate representation after cross-engine alignment
- Examples: `omniparse/models/consensus.py`
- Pattern: Contains aligned_texts dict (engine_name → [char_list]) and per-engine confidence scores
- Used as input to consensus voting
- Purpose: Finalized per-page output with resolved text and decision metadata
- Examples: `omniparse/models/consensus.py`
- Pattern: Contains list of resolved regions with source (identical/voting/voting_fallback/arbitration/hitl_fallback), confidence, and CE value
- Input to Markdown compiler
- Purpose: Structured audit trail for compliance and monitoring
- Examples: `omniparse/models/pipeline.py`
- Pattern: PageLog objects (per-page metrics) + ProcessingLog aggregates (totals, rates, engine versions)
- RegionLog captures decision source, LLM invocation, HITL flag for every region
## Entry Points
- Location: `omniparse/app.py` → `hitl_web_app()` function (ASGI endpoint)
- Also: API router at `/submit`, `/status`, `/result` (created in `omniparse/api/router.py`)
- Triggers: HTTP POST to /submit with UploadFile, API key, optional callback_url
- Responsibilities: Auth, cost guard, async spawn, job tracking, webhook scheduling
- Location: `omniparse/pipeline.py` → `class Pipeline` (Modal @app.cls method)
- Callable: `modal.Cls.from_name("omniparse", "Pipeline")().process.remote(file_bytes, filename, budget_usd)`
- Triggers: Python client invocation via Modal
- Responsibilities: All orchestration logic (preprocessing through markdown compilation)
- Location: `omniparse/app.py` → `run_regression_suite()` function
- Triggers: Modal Cron schedule (every Monday 6am UTC)
- Responsibilities: Benchmark processing, baseline comparison, regression alerting
- Location: `omniparse/app.py` → `hitl_web_app()` (ASGI)
- Serves: Jinja2 templates for human OCR review
- Connects to: Modal Dict "omniparse-hitl" for flagged region storage
## Error Handling
- **Engine failures:** If an engine times out or crashes, its output is skipped. Consensus proceeds with remaining engines. Logged in ProcessingLog.
- **Preprocessing errors:** Corrupted pages have PagePayload.error set. Pipeline skips them, returns empty markdown with page_count=0.
- **Alignment failures:** If IoU matching produces no matches for a region, it's assigned source="single_engine" and passed through.
- **LLM rejection:** If arbiter output fails edit distance guard (hallucination), voting_fallback persists. HITL flag set.
- **Budget exceeded:** Cost guard raises BudgetExceededError at /submit time. Returns 429 to client. No pipeline execution.
- **API key invalid:** Auth layer raises ValueError. Router returns 401.
## Cross-Cutting Concerns
- Framework: Python logging module (stdlib)
- Levels: DEBUG (per-region alignment), INFO (page progress, specialist dispatch), WARNING (quality check low confidence), ERROR (engine failure, LLM hallucination detected)
- Pattern: Logger instantiated per module, logged at decision points (consensus source, CE value, LLM invocation)
- Pydantic BaseModel for all data contracts (Region, EngineOutput, AlignedRegion, ConsensusResult, PipelineResult)
- Element type whitelist: VALID_ELEMENT_TYPES in `omniparse/models/region.py`
- Bounding box validation: [x1, y1, x2, y2] coordinates enforced as exactly 4 floats
- Confidence score range: [0.0, 1.0] enforced by Pydantic Field
- API key lookup in Modal Dict via `validate_api_key()` in `omniparse/api/auth.py`
- Returns structured entry with budget_usd, user metadata
- All budget enforcement via cost_guard module before pipeline execution
- Modal GPU pricing baked in: A10G $0.000306/sec, L4 $0.000164/sec, CPU $0.000043/sec
- Per-engine duration recorded in ProcessingLogBuilder
- Estimated cost_usd computed in finalize() (approximate, for tracking only)
- Budget exhaustion checked at /submit time via cost_guard
<!-- GSD:architecture-end -->

<!-- GSD:workflow-start source:GSD defaults -->
## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:
- `/gsd:quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd:debug` for investigation and bug fixing
- `/gsd:execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->



<!-- GSD:profile-start -->
## Developer Profile

> Profile not yet configured. Run `/gsd:profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
