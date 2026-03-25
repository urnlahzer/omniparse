# OmniParse

High-fidelity multi-engine OCR pipeline that converts challenging scanned documents into LLM-ready Markdown, deployed on [Modal](https://modal.com) serverless GPU infrastructure.

**Supported formats:** PDF, PNG, JPG, JPEG, TIFF

Runs a five-engine ensemble — pdfplumber, PaddleOCR, Docling (always), TrOCR, Dots.ocr (on-demand) — with spatial alignment, majority voting, and constrained LLM reconciliation.

## Quick Start

### Prerequisites

- Python 3.11+
- [Modal](https://modal.com) account (free Starter plan: $30/month credits)
- Modal CLI: `pip install modal`

### Local Setup

```bash
git clone https://github.com/urnlahzer/omniparse.git
cd omniparse
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Run tests (no GPU needed — all GPU code is mocked)
python -m pytest omniparse/tests/ -q
```

### Deploy to Modal

```bash
# Authenticate with Modal (opens browser)
modal token new

# Deploy the app — builds 7 container images
# First deploy takes 10-15 minutes (downloads ~5 GB of dependencies)
# Subsequent deploys take <5 seconds (images are cached)
modal deploy omniparse/deploy.py

# Download model weights into Modal Volume (~19 GB, runs on Modal servers)
# This takes 5-10 minutes on first run
modal run omniparse/setup_volume.py
```

#### First Deploy Notes

The first `modal deploy` builds container images for all engines. This involves downloading large packages:

| Image | Key packages | Size |
|-------|-------------|------|
| cpu | pdfplumber, FastAPI, Pillow | ~200 MB |
| paddleocr | PaddlePaddle-GPU 3.3.0, PaddleOCR 3.4+ | ~2 GB |
| docling | Docling, PyTorch, onnxruntime-gpu | ~3 GB |
| trocr | Transformers, PyTorch, PaddlePaddle | ~2.5 GB |
| dots | vLLM, PyTorch | ~2 GB |
| llm_arbiter | vLLM 0.17+, Qwen-VL-utils | ~2 GB |

**PaddlePaddle-GPU** downloads from `paddlepaddle.org.cn` (Baidu's official package index — the only distribution source for the GPU build). This is normal and expected.

After first build, all images are cached on Modal. Redeploys only upload changed source code (~seconds).

## Architecture

```
Document → Preprocess → ┌─ pdfplumber (CPU)  ─┐
                         ├─ PaddleOCR (A10G)  ─┤
                         └─ Docling (L4)      ─┘
                                │
                         Noise Filter ── removes line/page numbers, footer fragments
                                │
                         Docling Pre-Merge ── joins word-level fragments into lines
                                │
                         Alignment → Consensus → Markdown
                            ↑            ↑
                            │            │
                  ┌─ TrOCR (L4)  ─┐     │     LLM Arbiter
                  └─ Dots.ocr (A10G) ─┘  │   (Qwen3-VL-8B)
                       (on-demand)        │
                                          │
                  Smart Dispatch ─────────┘
                  (PaddleOCR labels → specialist engines)
```

### Always-Run Engines (every page)
- **pdfplumber** (CPU): Embedded text extraction with character-level bounding boxes
- **PaddleOCR PP-StructureV3** (A10G GPU): Layout classification + OCR text extraction
- **Docling** (L4 GPU): Hierarchical structure, table recovery, reading order

### On-Demand Engines (specialist regions only)
- **TrOCR** (L4 GPU): Handwriting recognition with DBNet line segmentation
- **Dots.ocr** (A10G GPU): Formula-to-LaTeX, chart-to-SVG

### Pre-Alignment Processing
- **Noise Filter**: Removes formatting artifacts before cross-engine matching — court line numbers in the left margin, page numbers at page bottom, and footer fragments. Prevents layout noise from inflating single-engine counts.
- **Docling Pre-Merge**: Joins word-level Docling fragments that sit on the same text line into single regions. Docling produces word-level bounding boxes where PaddleOCR detects full lines; without merging, the granularity mismatch causes orphans during IoU alignment. Uses union-find for transitive merging with conservative thresholds (vertical overlap + horizontal gap < 5% of page width).

### Consensus Pipeline
1. **IoU Alignment**: Match bounding boxes across engines (six-stage pipeline, see [Design Notes](#six-stage-matching-pipeline-and-why-each-stage-exists))
2. **Text Alignment**: Needleman-Wunsch sequence alignment via `sequence-align`
3. **Consensus Entropy**: Route low-CE to majority voting, high-CE to LLM arbitration
4. **LLM Arbitration**: Qwen3-VL-8B-Instruct FP8 with 3-layer hallucination defense
5. **Markdown Compilation**: GFM output with headers, tables, LaTeX, HITL flags

## Usage

### SDK (from another Modal app or Python)

```python
import modal

Pipeline = modal.Cls.from_name("omniparse", "Pipeline")
pipeline = Pipeline()

with open("document.pdf", "rb") as f:
    result = pipeline.process.remote(f.read(), "document.pdf")

print(result["markdown"])       # GFM Markdown output
print(result["processing_log"]) # Full audit trail
print(result["metadata"])       # Page count, duration, cost estimate
```

#### Budget enforcement

Pass `budget_usd` to cap per-job spending. Processing stops after the page that exceeds the budget and returns partial results:

```python
result = pipeline.process.remote(file_bytes, "large.pdf", budget_usd=0.50)
print(result["metadata"]["budget_exceeded"])  # True if capped early
```

#### Async (fire-and-forget)

```python
call = pipeline.process.spawn(file_bytes, "document.pdf")
# ... do other work ...
result = call.get(timeout=120)
```

### HTTP API

```bash
# Synchronous — process and return result
curl -X POST https://<your-namespace>--omniparse-parse-document.modal.run \
  -F "file=@document.pdf"
```

### Async Job API

For long-running documents, use the async job endpoints. All endpoints require an `X-API-Key` header.

```bash
# Submit a document (returns job_id immediately)
curl -X POST https://<your-namespace>--omniparse-api.modal.run/submit \
  -H "X-API-Key: <your-key>" \
  -F "file=@document.pdf"

# Optional form fields:
#   callback_url  — webhook URL for completion notification
#   ce_threshold  — consensus entropy cutoff (default 0.4)
#   confidence_floor — minimum confidence to accept (default 0.0)

# Poll job status
curl https://<your-namespace>--omniparse-api.modal.run/status/{job_id} \
  -H "X-API-Key: <your-key>"

# Retrieve result when complete
curl https://<your-namespace>--omniparse-api.modal.run/result/{job_id} \
  -H "X-API-Key: <your-key>"
```

When `callback_url` is provided, a webhook is fired on job completion with an HMAC-SHA256 signature (signed with `OMNIPARSE_WEBHOOK_SECRET`) in the `X-Webhook-Signature` header.

### HITL Review Interface

After deployment, a web UI for reviewing flagged OCR regions is available at:
```
https://<your-namespace>--omniparse-hitl-web-app.modal.run
```

## Output Format

GFM Markdown with YAML frontmatter:

```markdown
---
title: document.pdf
pages: 10
processed: "2026-03-20T15:30:00Z"
hitl_flags: 2
---

# Section Header

Regular paragraph text extracted from the document.

| Column 1 | Column 2 |
|----------|----------|
| Data     | Values   |

$$E = mc^2$$

*handwritten annotation* <!-- handwritten -->

<!-- REVIEW NEEDED: [100,200,500,300] page=3 confidence=0.35 -->
```

## Project Structure

```
omniparse/
├── app.py                  # Modal App, container images, HITL/cron
├── deploy.py               # Deploy entrypoint (imports all modules)
├── setup_volume.py         # One-time model weight download
├── pipeline.py             # End-to-end orchestrator
├── preprocess.py           # DPI normalization, deskew, PDF chunking
├── alignment.py            # IoU matching, Needleman-Wunsch alignment
├── consensus.py            # CE computation, voting, arbitration wiring
├── llm_arbiter.py          # Qwen3-VL-8B with hallucination safeguards
├── markdown_compiler.py    # GFM Markdown output
├── dispatch.py             # Smart routing to specialist engines
├── normalization.py        # Bounding box coordinate conversion
├── quality_check.py        # pdfplumber ground truth detection
├── noise_filter.py         # Pre-alignment removal of line/page numbers, footers
├── docling_premerge.py     # Join word-level Docling fragments into lines
├── wbf.py                  # Weighted Boxes Fusion (vendored, MIT)
├── type_compatibility.py   # Type compatibility utilities
├── observability.py        # Processing log builder, cost estimation
├── engines/
│   ├── pdfplumber_engine.py
│   ├── paddleocr_engine.py
│   ├── docling_engine.py
│   ├── trocr_engine.py
│   └── dots_engine.py
├── models/
│   ├── region.py           # Region, EngineOutput schemas
│   ├── consensus.py        # AlignedRegion, ConsensusResult
│   ├── pipeline.py         # PipelineResult, ProcessingLog
│   └── page.py             # PagePayload schema
├── api/
│   ├── router.py           # Async job API (/submit, /status, /result)
│   ├── auth.py             # API key validation
│   ├── webhooks.py         # HMAC-SHA256 signed webhook delivery
│   └── cost_guard.py       # Per-job budget enforcement
├── hitl/
│   ├── router.py           # Review UI endpoints (FastAPI)
│   ├── models.py           # HITL data schemas
│   └── templates/          # Jinja2 templates (list, detail views)
├── regression/             # Benchmark runner + baselines
└── tests/                  # 498 tests across 31 files, all mocked (no GPU needed)
```

## Testing

All tests run locally without GPU or Modal credentials:

```bash
# Quick run
python -m pytest omniparse/tests/ -q

# Verbose with coverage
python -m pytest omniparse/tests/ -v --tb=long
```

GPU engines are tested via mock objects. Modal configuration is verified via source inspection.

## Environment Variables

Copy `example.env` to `.env` and configure:

| Variable | Required | Description |
|----------|----------|-------------|
| `OMNIPARSE_SAMPLES_DIR` | No | Path to sample PDFs for integration tests |
| `OMNIPARSE_WEBHOOK_SECRET` | Yes (prod) | Shared secret for HMAC-SHA256 webhook signing (defaults to `"default-secret"` — **set a real value in production**) |

### API Key Management

The async API authenticates requests via `X-API-Key` header. Keys are stored in a Modal Dict (`omniparse-api-keys`). Each key tracks cumulative spend against an optional `budget_usd` cap — requests that would exceed the budget are rejected with HTTP 402.

## Cost

On Modal Starter plan ($30/month free credits):

| Resource | Rate | Typical usage |
|----------|------|---------------|
| A10G GPU | ~$1.10/hr | PaddleOCR, Dots.ocr, LLM arbiter |
| L4 GPU | ~$0.59/hr | Docling, TrOCR |
| CPU | ~$0.16/hr | pdfplumber, preprocessing, compilation |

Estimated amortized cost: <$0.008 per page at scale.

All engines scale to zero when idle (`min_containers=0`). Use the `warm_engines()` function before batch processing to pre-warm containers.

## Design Notes

This section documents hard-won lessons from building the multi-engine ensemble. These are things that aren't obvious from the code alone and would otherwise be lost.

### The Coordinate System Problem

The single biggest obstacle to cross-engine alignment was bounding box coordinate systems. Each engine outputs coordinates differently:

- **pdfplumber**: PDF points (72 DPI), top-left origin
- **PaddleOCR**: Pixel coordinates (300 DPI), top-left origin
- **Docling**: PDF points (72 DPI), **bottom-left** origin (Y-axis flipped)

Without normalization, IoU between engines was essentially zero — pdfplumber boxes were ~4x smaller than PaddleOCR boxes in the same space. Multi-engine voting was 0% for the first deployed version.

**Solution**: Normalize all coordinates to `[0,1]` page-fraction space before any spatial matching. This makes alignment resolution-independent and engine-agnostic. The normalization module (`normalization.py`) converts pixel coordinates to unit space; alignment reads `bounding_box_norm` when available and falls back to raw `bounding_box`.

**Lesson**: If you're combining spatial data from multiple sources, normalize to a unit coordinate space immediately. Don't try to convert between source-specific systems.

### Six-Stage Matching Pipeline (and Why Each Stage Exists)

The alignment pipeline went through several iterations. A single IoU threshold doesn't work because engines disagree on bounding box boundaries in different ways. The final six-stage pipeline exists because each stage catches a specific failure mode the previous stages miss:

1. **IoU matching** (threshold >0.5): Catches well-aligned boxes. The threshold started at 0.85 (from object detection literature) but real cross-engine variance is much higher than same-model variance — 0.5 was the practical sweet spot. Lowering to 0.3 caused false matches.

2. **Center-distance rescue** (center distance ≤0.05, IoU >0.05): PaddleOCR and Docling often detect the same region with a 40-100px Y-axis offset on scanned documents. For small regions (~60px tall), IoU drops below 0.5 despite obvious visual correspondence. Center-distance catches these — but proximity alone caused false matches on stacked text lines (this was reverted once before adding the IoU floor requirement).

3. **Containment ratio** (threshold ≥0.6): Handles cross-granularity differences where one engine detects a paragraph as one box and another splits it into lines. The threshold started at 0.7 but a real document pair had CR=0.6999, so 0.6 catches these near-misses. Many-to-one merging concatenates contained regions in reading order.

4. **Weighted Boxes Fusion**: Groups partially overlapping boxes using weighted averaging of coordinates. Vendored from ensemble-boxes with added provenance tracking so vote counts map back to source engines.

5. **Agglomerative clustering** (scipy complete-linkage, distance threshold 0.92): Last-resort grouping for orphans with marginal overlap (IoU ~0.08-0.5). Uses `1 - IoU` as the distance metric.

6. **Orphan fallback**: Anything still unmatched becomes a single-engine region.

**Lesson**: No single spatial matching technique works across all the ways engines disagree. A layered approach with a "consumed set" (preventing double-matching) is more robust than tuning a single threshold.

### What Didn't Work

**NW gap character collision**: The Needleman-Wunsch text alignment used `_` as the gap character. This crashed on OCR text containing underscores (which is common in legal documents). Changed to null byte `\x00`.

**Pairwise alignment length assumption**: With 3 engines, NW alignment is computed pairwise (A-B, A-C, B-C). Each pair can produce a different-length aligned output. Both consensus entropy and majority voting initially assumed equal lengths, causing `IndexError` on real documents. Required explicit padding/truncation handling.

**Docling word-level granularity**: Docling produces word-level bounding boxes where PaddleOCR detects full lines. Without pre-merging, Docling fragments created massive orphan rates during IoU alignment. The conservative pre-merge step (union-find with vertical overlap + horizontal gap <5% page width) was required to make Docling regions usable in the ensemble.

**Modal serialization quirk**: Docling results were silently discarded for weeks because Modal serialized dictionary keys as integers but the pipeline looked them up as strings. No error was raised — just empty Docling output on every page.

**Docling on image-only PDFs**: Docling's `get_bitmap_rects` crashes on pages that are pure images (common in scanned documents). Fixed by enabling `force_full_page_ocr=True`, which also makes Docling an independent OCR engine rather than relying on embedded text.

**PaddleOCR dependency chain**: PP-StructureV3 requires `paddlex[ocr]` extra, not just `paddleocr`. Installing only `paddleocr` pulls `paddlex` without OCR sub-dependencies, causing silent failures. Additionally, `paddleocr` pulls `opencv-contrib-python` (non-headless) which requires `libGL` — solved by force-reinstalling `opencv-python-headless` after PaddleOCR.

**PaddleOCR startup check**: PaddleOCR/PaddleX runs a connectivity check against Chinese servers on startup. This adds ~10s to cold starts in non-China regions. Disabled via `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=1`.

### Key Design Decisions

**Scale-to-zero everything** (`min_containers=0`): All GPU engines scale to zero when idle. This means zero cost when not processing but ~30-60s cold start on first request. The `warm_engines()` function pre-spawns containers before batch jobs. An early version kept PaddleOCR at `min_containers=1` for faster response, but this costs $1.10/hr idle.

**Bake models into container images**: First versions downloaded models from HuggingFace on every cold start (~60s overhead). Moving model downloads into the image build step (via `huggingface_hub.snapshot_download`) eliminates this. Models refresh automatically on redeploy; bump `_PADDLE_MODEL_REV` or `_DOCLING_MODEL_REV` to force a cache bust.

**Per-engine container images**: Each engine gets its own Docker image with exact dependencies. PaddleOCR needs PaddlePaddle-GPU + libGL; Docling needs PyTorch + onnxruntime-gpu; Dots.ocr needs vLLM. Sharing images would create version conflicts and bloated containers.

**Dependency injection for testing**: The pipeline accepts an `engines=` parameter for mock injection. All 498 tests run locally without GPU or Modal credentials. GPU engines are tested via mock objects that return realistic `EngineOutput` dicts. Modal configuration is verified via source inspection (reading `app.py` as Python AST).

**Three-layer LLM hallucination defense**: The LLM arbiter (Qwen3-VL-8B) resolves high-entropy disagreements between engines, but LLMs can hallucinate content that looks plausible but was never in the document. Three independent checks catch this: (1) edit distance — reject if output is far from all engine candidates, (2) consecutive insertion detection — reject if >5 consecutive characters appear that weren't in any candidate, (3) legal-field regex — flag if novel dollar amounts, dates, citations, or statute numbers appear. If the LLM output is rejected, the region falls back to majority vote and gets flagged for human review.

**Pre-alignment noise filtering**: Court documents have line numbers in the left margin, page numbers at the bottom, and footer fragments. These are real text that engines correctly detect, but they inflate single-engine orphan counts and confuse spatial matching. Filtering them before alignment (not after) was critical — otherwise they'd create spurious cross-engine "disagreements."

**Ground truth detection**: When pdfplumber and PaddleOCR agree >90% (measured by character-level Levenshtein), the PDF has clean embedded text and pdfplumber is treated as ground truth with 2x voting weight. This avoids unnecessary LLM arbitration on born-digital documents.

### Metrics Baseline

Measured on synthetic test scenarios as of the latest development phase:

| Scenario | Multi-engine resolution rate | Notes |
|----------|------------------------------|-------|
| Born-digital legal | 100% | pdfplumber ground truth, no arbitration needed |
| Type disagreement | 100% | Cross-type compatibility groups resolve header/printed_text |
| Scanned legal | ~62% | Improved from 28.4% through alignment layering |
| Degraded scanned | ~29% | Many orphans due to severe degradation |
| Mixed (printed + handwriting) | varies | Specialist dispatch to TrOCR/Dots.ocr |

Target accuracy thresholds:
- CER <1% on clean digital legal
- Table TEDS >95% on degraded scanned
- CER <5% on handwriting samples
- LLM invocation rate <5% on born-digital, <15% on scanned

## License

Apache 2.0
