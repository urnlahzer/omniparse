"""End-to-end pipeline orchestrator -- wires preprocessing, engines, alignment,
consensus, arbitration, and Markdown compilation into a single concurrent flow.

Entry points:
- Pipeline.process (SDK): modal.Cls.from_name("omniparse", "Pipeline")
- parse_document (HTTP): POST multipart file upload via @modal.fastapi_endpoint
- warm_engines: Pre-warm engine containers before batch dispatch

All pure-function logic (alignment, consensus, compilation) runs locally inside
the orchestrator -- only engine invocations are Modal remote calls.
"""
import logging
import time
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import modal

try:
    from fastapi import UploadFile
except ImportError:  # pragma: no cover -- fastapi only in Modal container image
    UploadFile = None  # type: ignore[assignment,misc]

from omniparse.app import app, cpu_image
from omniparse.models.region import EngineOutput
from omniparse.models.consensus import ConsensusResult
from omniparse.normalization import normalize_to_unit
from omniparse.noise_filter import filter_noise_regions
from omniparse.docling_premerge import premerge_docling_regions
from omniparse.models.pipeline import PipelineResult, ProcessingLog
from omniparse.observability import ProcessingLogBuilder
from omniparse.api.cost_guard import check_budget, BudgetExceededError

logger = logging.getLogger(__name__)


def _default_engines():
    """Load real Modal engine callables. Separated for dependency injection."""
    from omniparse.engines.pdfplumber_engine import run_pdfplumber
    from omniparse.engines.paddleocr_engine import PaddleOCREngine
    from omniparse.engines.docling_engine import DoclingEngine
    return {
        "run_pdfplumber": run_pdfplumber,
        "PaddleOCREngine": PaddleOCREngine,
        "DoclingEngine": DoclingEngine,
    }


class _ModalArbiterAdapter:
    """Adapt Modal LLMArbiter proxy so .run()/.run_handwriting() work as plain calls.

    Modal @app.cls methods are Function objects that require .remote().
    consensus.arbitrate_page expects a duck-typed arbiter with plain .run().
    This adapter bridges the two so consensus.py stays testable with mocks.
    """

    def __init__(self, modal_arbiter):
        self._arbiter = modal_arbiter

    def run(self, *, image_bytes: bytes, candidates: dict) -> dict:
        return self._arbiter.run.remote(image_bytes=image_bytes, candidates=candidates)


def process_document(
    file_bytes: bytes,
    filename: str,
    log_builder: ProcessingLogBuilder,
    *,
    engines: Optional[dict] = None,
    budget_usd: float | None = None,
    ce_threshold: float | None = None,
    confidence_floor: float | None = None,
) -> PipelineResult:
    """Core pipeline logic -- called from both HTTP and SDK entry points.

    Orchestrates: preprocess -> concurrent 3-engine dispatch -> specialist
    routing -> alignment -> consensus -> arbitration -> Markdown compilation.

    Args:
        file_bytes: Raw document bytes (PDF, PNG, JPG, JPEG, TIFF).
        filename: Original filename with extension.
        log_builder: Pre-started ProcessingLogBuilder instance.
        engines: Optional dict of engine callables for dependency injection.
            When None (production), loads real Modal engines via _default_engines().
            When provided (testing), uses injected callables directly.

    Returns:
        PipelineResult with markdown, processing_log, and metadata.
    """
    if engines is None:
        engines = _default_engines()
    run_pdfplumber = engines["run_pdfplumber"]
    PaddleOCREngine = engines["PaddleOCREngine"]
    DoclingEngine = engines["DoclingEngine"]

    # -----------------------------------------------------------------
    # 1. Preprocess
    # -----------------------------------------------------------------
    from omniparse.preprocess import preprocess

    pages = preprocess(file_bytes, filename)
    valid_pages = [p for p in pages if p.error is None]

    if not valid_pages:
        log = log_builder.finalize()
        return PipelineResult(
            markdown="",
            processing_log=log,
            metadata={
                "page_count": 0,
                "filename": filename,
                "processing_time_s": log.total_duration_s,
                "hitl_flag_count": 0,
            },
        )

    is_pdf = filename.lower().endswith(".pdf")

    # -----------------------------------------------------------------
    # 2. Parallel always-run engine dispatch (PARA-01)
    # -----------------------------------------------------------------
    def _run_pdfplumber_batch():
        if not is_pdf:
            return [
                EngineOutput(page=p.page_num, engine="pdfplumber", regions=[]).model_dump()
                for p in valid_pages
            ], 0.0
        t0 = time.perf_counter()
        results = list(run_pdfplumber.map(
            [p.pdf_bytes for p in valid_pages],
            [p.page_num for p in valid_pages],
        ))
        return results, time.perf_counter() - t0

    def _run_paddleocr_batch():
        t0 = time.perf_counter()
        paddle = PaddleOCREngine()
        results = list(paddle.run.map(
            [p.image_bytes for p in valid_pages],
            [p.page_num for p in valid_pages],
        ))
        return results, time.perf_counter() - t0

    def _run_docling_batch():
        if not is_pdf:
            return {}, 0.0
        t0 = time.perf_counter()
        docling = DoclingEngine()
        results = docling.run.remote(
            pdf_bytes=valid_pages[0].pdf_bytes,
            page_heights={str(p.page_num): p.height for p in valid_pages},
        )
        return results, time.perf_counter() - t0

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(_run_pdfplumber_batch): "pdfplumber",
            executor.submit(_run_paddleocr_batch): "paddleocr",
            executor.submit(_run_docling_batch): "docling",
        }
        engine_results = {}
        for future in as_completed(futures):
            engine_name = futures[future]
            results, duration = future.result()
            engine_results[engine_name] = (results, duration)

    pdfplumber_results, pdfplumber_duration = engine_results["pdfplumber"]
    paddleocr_results, paddleocr_duration = engine_results["paddleocr"]
    docling_all, docling_duration = engine_results["docling"]

    # -----------------------------------------------------------------
    # 3. Per-page consensus pipeline
    # -----------------------------------------------------------------
    from omniparse.alignment import match_regions_across_engines, align_region_group
    from omniparse.consensus import resolve_page, arbitrate_page, crop_region_image
    from omniparse.quality_check import check_text_quality
    from omniparse.dispatch import classify_dispatch

    num_pages = len(valid_pages)
    all_consensus: list[ConsensusResult] = []
    cumulative_cost_usd = 0.0

    for i, page in enumerate(valid_pages):
        pn = page.page_num

        # a) Reconstruct EngineOutput objects from dicts (model_validate for nested coercion)
        plumber_eo = EngineOutput.model_validate(pdfplumber_results[i])
        paddle_eo = EngineOutput.model_validate(paddleocr_results[i])

        # Modal may return int or str keys depending on serialization format
        docling_dict = (
            (docling_all.get(pn) or docling_all.get(str(pn)))
            if isinstance(docling_all, dict) else None
        )
        docling_eo = (
            EngineOutput.model_validate(docling_dict)
            if docling_dict
            else EngineOutput(page=pn, engine="docling", regions=[])
        )

        # a2) [0,1] normalization (NORM-01, NORM-02): compute bounding_box_norm from pixel coords
        for eo in [plumber_eo, paddle_eo, docling_eo]:
            for region in eo.regions:
                region.bounding_box_norm = normalize_to_unit(
                    region.bounding_box, page.width, page.height
                )

        # a3) Filter noise regions (line numbers, page numbers, footer fragments)
        for eo in [plumber_eo, paddle_eo, docling_eo]:
            eo.regions = filter_noise_regions(eo.regions)

        # a4) Conservative Docling pre-merge (D-01): merge word-level fragments on same line
        docling_eo.regions = premerge_docling_regions(docling_eo.regions)

        # b) Record engine results in log_builder
        log_builder.start_page(pn, is_ground_truth=False)
        log_builder.record_engine_result(pn, "pdfplumber", pdfplumber_duration / max(1, num_pages), len(plumber_eo.regions))
        log_builder.record_engine_result(pn, "paddleocr", paddleocr_duration / max(1, num_pages), len(paddle_eo.regions))
        log_builder.record_engine_result(pn, "docling", docling_duration / max(1, num_pages), len(docling_eo.regions))

        # c) Quality check
        qc = check_text_quality(plumber_eo, paddle_eo)
        # Update ground truth status
        log_builder._pages[pn].is_ground_truth = qc["is_ground_truth"]

        # d) Specialist dispatch (AFTER PaddleOCR results available)
        dispatch = classify_dispatch(paddle_eo)
        engine_outputs = [plumber_eo, paddle_eo, docling_eo]

        has_specialists = (
            len(dispatch["trocr"]) > 0
            or len(dispatch["dots_formula"]) > 0
            or len(dispatch["dots_chart"]) > 0
        )

        if has_specialists:
            from omniparse.engines.trocr_engine import TrOCREngine
            from omniparse.engines.dots_engine import DotsEngine

            # TrOCR regions
            if dispatch["trocr"]:
                trocr = TrOCREngine()
                paddle_inst = PaddleOCREngine()
                for region in dispatch["trocr"]:
                    crop = crop_region_image(page.image_bytes, region.bounding_box)
                    trocr_result = trocr.run.remote(
                        region_image=crop,
                        region_bbox=region.bounding_box,
                        region_id=region.id,
                    )
                    trocr_eo = EngineOutput.model_validate(trocr_result)
                    for r in trocr_eo.regions:
                        r.bounding_box_norm = normalize_to_unit(
                            r.bounding_box, page.width, page.height
                        )
                    engine_outputs.append(trocr_eo)

                    # Second opinion from PP-OCRv5 handwriting
                    hw_result = paddle_inst.run_handwriting.remote(region_image=crop)
                    hw_eo = EngineOutput.model_validate(hw_result)
                    for r in hw_eo.regions:
                        r.bounding_box_norm = normalize_to_unit(
                            r.bounding_box, page.width, page.height
                        )
                    engine_outputs.append(hw_eo)

                log_builder.record_specialist_dispatch(pn, "trocr", len(dispatch["trocr"]))

            # Dots formula regions
            if dispatch["dots_formula"]:
                dots = DotsEngine()
                for region in dispatch["dots_formula"]:
                    crop = crop_region_image(page.image_bytes, region.bounding_box)
                    dots_result = dots.run_formula.remote(region_image=crop)
                    dots_eo = EngineOutput.model_validate(dots_result)
                    for r in dots_eo.regions:
                        r.bounding_box_norm = normalize_to_unit(
                            r.bounding_box, page.width, page.height
                        )
                    engine_outputs.append(dots_eo)
                log_builder.record_specialist_dispatch(pn, "dots_formula", len(dispatch["dots_formula"]))

            # Dots chart regions
            if dispatch["dots_chart"]:
                dots = DotsEngine()
                for region in dispatch["dots_chart"]:
                    crop = crop_region_image(page.image_bytes, region.bounding_box)
                    w = int(region.bounding_box[2] - region.bounding_box[0])
                    h = int(region.bounding_box[3] - region.bounding_box[1])
                    dots_result = dots.run_chart.remote(region_image=crop, width=w, height=h)
                    dots_eo = EngineOutput.model_validate(dots_result)
                    for r in dots_eo.regions:
                        r.bounding_box_norm = normalize_to_unit(
                            r.bounding_box, page.width, page.height
                        )
                    engine_outputs.append(dots_eo)
                log_builder.record_specialist_dispatch(pn, "dots_chart", len(dispatch["dots_chart"]))

        # e) Alignment
        for eo in engine_outputs:
            logger.info("Page %d engine %s: %d regions, bbox sample: %s",
                        pn, eo.engine, len(eo.regions),
                        eo.regions[0].bounding_box if eo.regions else "[]")
        matched = match_regions_across_engines(engine_outputs)
        aligned = [align_region_group(g) for g in matched]

        # f) Consensus
        consensus_kwargs = {"is_ground_truth": qc["is_ground_truth"]}
        if ce_threshold is not None:
            consensus_kwargs["ce_threshold"] = ce_threshold
        result = resolve_page(aligned, pn, **consensus_kwargs)

        # g) Arbitration (for high-CE regions)
        if any(r.needs_arbitration for r in result.regions):
            from omniparse.llm_arbiter import LLMArbiter
            arbiter = _ModalArbiterAdapter(LLMArbiter())
            result = arbitrate_page(result, arbiter, page.image_bytes)

        # g2) Confidence floor filtering -- flag low-confidence regions for HITL
        if confidence_floor is not None and confidence_floor > 0.0:
            flagged = []
            for region in result.regions:
                if region.confidence is not None and region.confidence < confidence_floor:
                    region = region.model_copy(update={"hitl_flag": True})
                flagged.append(region)
            result = result.model_copy(update={"regions": flagged})

        # h) Record all region decisions in log_builder
        for region in result.regions:
            log_builder.record_region_decision(
                pn,
                region.region_id,
                region.element_type,
                region.bounding_box,
                list(region.engine_texts.keys()),
                region.source,
                region.confidence,
                ce_value=None,
                llm_invoked=(region.source == "arbitration"),
                hitl_flag=region.hitl_flag,
            )

        # i) Estimate page cost and store ConsensusResult
        page_cost = log_builder.estimate_page_cost(pn)
        cumulative_cost_usd += page_cost
        all_consensus.append(result)

        # j) Per-page budget enforcement (PROD-01)
        try:
            check_budget(cumulative_cost_usd, budget_usd)
        except BudgetExceededError:
            logger.warning(
                "Budget exceeded after page %d: $%.4f >= $%.4f. "
                "Returning partial result (%d/%d pages).",
                pn, cumulative_cost_usd, budget_usd, i + 1, len(valid_pages),
            )
            break

    # -----------------------------------------------------------------
    # 4. Compile to Markdown
    # -----------------------------------------------------------------
    from omniparse.markdown_compiler import compile_document

    budget_exceeded = len(all_consensus) < len(valid_pages)
    markdown = compile_document(all_consensus)

    # -----------------------------------------------------------------
    # 5. Finalize and return
    # -----------------------------------------------------------------
    log = log_builder.finalize()
    metadata = {
        "page_count": len(all_consensus),
        "total_page_count": len(valid_pages),
        "filename": filename,
        "processing_time_s": log.total_duration_s,
        "hitl_flag_count": log.total_hitl_flags,
        "budget_exceeded": budget_exceeded,
    }
    return PipelineResult(markdown=markdown, processing_log=log, metadata=metadata)


# --------------------------------------------------------------------------
# Pre-warming function (INFR-05)
# --------------------------------------------------------------------------

@app.function(image=cpu_image, timeout=60)
def warm_engines():
    """Pre-warm engine containers by triggering lightweight spawn calls.

    Call ~30 seconds before submitting a batch job. Each .spawn()
    triggers container startup (including @modal.enter model loading)
    without blocking on results.
    """
    from omniparse.engines.paddleocr_engine import PaddleOCREngine
    from omniparse.engines.docling_engine import DoclingEngine

    # Spawn lightweight calls to trigger container startup
    # PaddleOCR already has min_containers=1 but ping ensures warm
    paddle = PaddleOCREngine()
    paddle.run.spawn(b"", 0)  # empty bytes triggers fast return

    docling = DoclingEngine()
    docling.run.spawn(b"", {})  # empty bytes triggers fast return

    logger.info("Engine pre-warming initiated")


# --------------------------------------------------------------------------
# SDK entry point (INTG-02)
# --------------------------------------------------------------------------

@app.cls(
    image=cpu_image,
    timeout=900,  # 15 minutes: accommodates engine cold starts + processing
    min_containers=0,
    max_containers=5,
)
class Pipeline:
    """End-to-end document processing pipeline.

    SDK callers use:
      cls = modal.Cls.from_name("omniparse", "Pipeline")
      instance = cls()
      result = instance.process.remote(file_bytes, "document.pdf")
      # -- OR async --
      call = instance.process.spawn(file_bytes, "document.pdf")
      result = call.get(timeout=120)
    """

    @modal.method()
    def process(
        self,
        file_bytes: bytes,
        filename: str,
        budget_usd: float | None = None,
        ce_threshold: float | None = None,
        confidence_floor: float | None = None,
    ) -> dict:
        """Process a document through the full OmniParse pipeline.

        Args:
            file_bytes: Raw document bytes (PDF, PNG, JPG, JPEG, TIFF).
            filename: Original filename with extension.
            budget_usd: Optional per-job budget limit in USD. When set,
                processing stops after any page where cumulative cost
                exceeds this value. Partial results are returned.
            ce_threshold: Optional cross-entropy threshold for consensus.
                When set, overrides the default CE_THRESHOLD (0.4) in the
                consensus resolver. Lower values are stricter (more regions
                escalated to LLM arbitration).
            confidence_floor: Optional minimum confidence to accept a region.
                Regions below this confidence are flagged for HITL review.

        Returns:
            dict: PipelineResult.model_dump() with keys:
              - markdown: GFM Markdown string
              - processing_log: Full per-region decision trail
              - metadata: {page_count, total_page_count, filename, processing_time_s,
                           hitl_flag_count, budget_exceeded}
        """
        log_builder = ProcessingLogBuilder()
        log_builder.start()
        result = process_document(
            file_bytes, filename, log_builder,
            budget_usd=budget_usd,
            ce_threshold=ce_threshold,
            confidence_floor=confidence_floor,
        )
        return result.model_dump()


# --------------------------------------------------------------------------
# HTTP entry point (INTG-01)
# --------------------------------------------------------------------------

@app.function(image=cpu_image, timeout=600)
@modal.fastapi_endpoint(method="POST")
async def parse_document(file: "UploadFile") -> dict:
    """HTTP endpoint for document processing.

    Accepts multipart file upload, returns PipelineResult JSON.
    Public URL provided by Modal after deployment.

    Example:
      curl -X POST https://<modal-url>/parse_document \\
        -F "file=@document.pdf"
    """
    file_bytes = await file.read()
    filename = file.filename or "document.pdf"
    pipeline = Pipeline()
    return pipeline.process.remote(file_bytes, filename)


# --------------------------------------------------------------------------
# Webhook notifier (PROD-02)
# --------------------------------------------------------------------------

@app.function(image=cpu_image, timeout=120)
async def notify_on_complete(job_id: str, callback_url: str):
    """Wait for job completion, then fire webhook to callback_url.

    Spawned by submit endpoint when callback_url is provided.
    Polls FunctionCall until complete, then delivers webhook.
    """
    import os

    from omniparse.api.webhooks import deliver_webhook
    from omniparse.api.models import WebhookPayload

    jobs_dict = modal.Dict.from_name("omniparse-jobs", create_if_missing=True)

    # Wait for the pipeline job to complete
    fc = modal.FunctionCall.from_id(job_id)
    try:
        result = fc.get(timeout=600)
        # Update job status in Dict
        job_meta = jobs_dict[job_id]
        job_meta["status"] = "completed"
        job_meta["result"] = result
        jobs_dict[job_id] = job_meta

        # Build and deliver webhook
        payload = WebhookPayload(
            job_id=job_id,
            status="completed",
            result_url=f"/result/{job_id}",
            summary={
                "page_count": result.get("metadata", {}).get("page_count", 0),
                "hitl_flag_count": result.get("metadata", {}).get("hitl_flag_count", 0),
                "total_cost_usd": result.get("processing_log", {}).get("total_cost_usd", 0),
                "processing_time_s": result.get("metadata", {}).get("processing_time_s", 0),
            },
        ).model_dump()

        # Get webhook secret from Modal Secret -- fail loudly if unset
        secret = os.environ.get("OMNIPARSE_WEBHOOK_SECRET")
        if not secret:
            raise RuntimeError("OMNIPARSE_WEBHOOK_SECRET must be set")
        await deliver_webhook(callback_url, payload, secret)

    except Exception as e:
        job_meta = jobs_dict.get(job_id, {})
        job_meta["status"] = "failed"
        job_meta["error"] = str(e)
        jobs_dict[job_id] = job_meta
