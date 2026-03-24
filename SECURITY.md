# Security

OmniParse is a multi-engine OCR ensemble pipeline running on Modal serverless infrastructure. This document describes accepted security risks, mitigations, and how to report vulnerabilities.

## Accepted Risks

### `trust_remote_code=True` in Model Loading

**What:** Two modules pass `trust_remote_code=True` when loading ML models via Hugging Face transformers / vLLM:

- **`omniparse/llm_arbiter.py`** (line ~392) -- Loading Qwen3-VL-8B-Instruct-FP8 via vLLM for LLM arbitration
- **`omniparse/engines/dots_engine.py`** (line ~184) -- Loading Dots.ocr-1.5 via vLLM for formula/chart recognition

This flag allows model repositories to include custom Python code that executes during model loading.

**Why accepted:** Both models are loaded from pinned local paths in the Modal Volume `/models/` directory, pre-cached during image build. No runtime downloads from external sources occur.

**Mitigations:**

- Model versions are pinned in Modal image definitions (not pulled dynamically at runtime)
- Models are loaded at container startup via `@modal.enter()` (build-time-only execution path)
- Modal container isolation provides sandboxing between workloads -- a compromised model affects only its own container class
- Model weights are stored on a private Modal Volume, not fetched from public URLs at runtime

**Residual risk:** If the Modal Volume is compromised or if a model update introduces malicious code, the container running that model class would be affected. This risk is bounded by Modal's per-container isolation and the fact that model updates require an explicit image rebuild and redeploy.

## Reporting Security Issues

If you discover a security vulnerability in OmniParse, please report it responsibly.

**Contact:** Email security@PLACEHOLDER.example.com

Please do not open public GitHub issues for security vulnerabilities. We will acknowledge receipt within 48 hours and provide an estimated timeline for a fix.
