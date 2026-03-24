"""Processing log builder -- accumulates per-region decisions during pipeline execution.

The ProcessingLogBuilder is instantiated at job start, accumulates engine timings,
region-level decisions, and specialist dispatch counts as the pipeline runs, then
produces a finalized ProcessingLog at job completion.

Cost estimation uses published Modal GPU pricing (approximate, for tracking):
- A10G: $1.10/hr = $0.000306/sec (PaddleOCR, Dots.ocr, LLM arbiter)
- L4: $0.59/hr = $0.000164/sec (Docling, TrOCR)
- CPU: $0.155/hr = $0.000043/sec (pdfplumber, preprocessing)
"""
import time

from omniparse.models.pipeline import ProcessingLog, PageLog, RegionLog

# GPU cost per second by engine (approximate Modal pricing)
GPU_COST_PER_SEC: dict[str, float] = {
    "paddleocr": 0.000306,   # A10G
    "docling": 0.000164,     # L4
    "trocr": 0.000164,       # L4
    "dots": 0.000306,        # A10G
    "llm_arbiter": 0.000306, # A10G
}
CPU_COST_PER_SEC: float = 0.000043

# Engine versions for reproducibility tracking
ENGINE_VERSIONS: dict[str, str] = {
    "pdfplumber": "0.11.9",
    "paddleocr": "PP-StructureV3",
    "docling": "2.80+",
    "trocr": "TrOCR-large-handwritten",
    "dots": "DotsOCR-1.5",
    "llm_arbiter": "Qwen3-VL-8B-Instruct-FP8",
}


class _PageAccumulator:
    """Internal accumulator for a single page's processing data."""

    def __init__(self, page_num: int, is_ground_truth: bool) -> None:
        self.page_num = page_num
        self.is_ground_truth = is_ground_truth
        self.engine_durations: dict[str, float] = {}
        self.regions: list[RegionLog] = []
        self.specialist_dispatches: dict[str, int] = {}

    @property
    def region_count(self) -> int:
        return len(self.regions)

    @property
    def llm_invocation_count(self) -> int:
        return sum(1 for r in self.regions if r.llm_invoked)

    @property
    def hitl_escalation_count(self) -> int:
        return sum(1 for r in self.regions if r.hitl_flag)


class ProcessingLogBuilder:
    """Accumulator for pipeline processing logs.

    Usage:
        builder = ProcessingLogBuilder()
        builder.start()
        builder.start_page(0, is_ground_truth=True)
        builder.record_engine_result(0, "pdfplumber", 0.5, 3)
        builder.record_region_decision(0, "r_001", "printed_text", [...], ...)
        log = builder.finalize()
    """

    def __init__(self) -> None:
        self._pages: dict[int, _PageAccumulator] = {}
        self._start_time: float = 0.0

    def start(self) -> None:
        """Record job start time."""
        self._start_time = time.perf_counter()

    def start_page(self, page_num: int, is_ground_truth: bool) -> None:
        """Begin accumulating data for a new page."""
        self._pages[page_num] = _PageAccumulator(page_num, is_ground_truth)

    def record_engine_result(
        self, page_num: int, engine: str, duration_s: float, region_count: int
    ) -> None:
        """Record an engine's processing duration for a page."""
        acc = self._pages[page_num]
        acc.engine_durations[engine] = duration_s

    def record_region_decision(
        self,
        page_num: int,
        region_id: str,
        element_type: str,
        bounding_box: list[float],
        engines_ran: list[str],
        resolution: str,
        confidence: float,
        ce_value: float | None = None,
        llm_invoked: bool = False,
        hitl_flag: bool = False,
    ) -> None:
        """Record a per-region consensus decision."""
        acc = self._pages[page_num]
        acc.regions.append(
            RegionLog(
                region_id=region_id,
                element_type=element_type,
                bounding_box=bounding_box,
                engines_ran=engines_ran,
                resolution=resolution,
                confidence=confidence,
                ce_value=ce_value,
                llm_invoked=llm_invoked,
                hitl_flag=hitl_flag,
            )
        )

    def record_specialist_dispatch(
        self, page_num: int, engine_type: str, count: int
    ) -> None:
        """Record specialist engine dispatch counts for a page."""
        acc = self._pages[page_num]
        acc.specialist_dispatches[engine_type] = (
            acc.specialist_dispatches.get(engine_type, 0) + count
        )

    def estimate_page_cost(self, page_num: int) -> float:
        """Compute estimated cost for a page from engine durations and rate constants.

        GPU engines use their published rate; unrecognized engines default to CPU rate.
        Returns the cost in USD.
        """
        acc = self._pages[page_num]
        cost = 0.0
        for engine, duration in acc.engine_durations.items():
            rate = GPU_COST_PER_SEC.get(engine, CPU_COST_PER_SEC)
            cost += duration * rate
        acc_page_cost = cost
        return acc_page_cost

    def finalize(self) -> ProcessingLog:
        """Produce the finalized ProcessingLog with computed totals.

        Estimates page costs, aggregates all page data, and computes job-level
        totals including LLM invocation rate.
        """
        total_duration_s = time.perf_counter() - self._start_time

        # Build PageLog objects with estimated costs
        page_logs: list[PageLog] = []
        for page_num in sorted(self._pages.keys()):
            acc = self._pages[page_num]
            estimated_cost = self.estimate_page_cost(page_num)
            page_logs.append(
                PageLog(
                    page_num=acc.page_num,
                    is_ground_truth=acc.is_ground_truth,
                    engine_durations=acc.engine_durations,
                    region_count=acc.region_count,
                    regions=list(acc.regions),
                    llm_invocation_count=acc.llm_invocation_count,
                    hitl_escalation_count=acc.hitl_escalation_count,
                    specialist_dispatches=dict(acc.specialist_dispatches),
                    estimated_cost_usd=estimated_cost,
                )
            )

        total_cost_usd = sum(p.estimated_cost_usd for p in page_logs)
        total_llm_invocations = sum(p.llm_invocation_count for p in page_logs)
        total_hitl_flags = sum(p.hitl_escalation_count for p in page_logs)
        total_regions = sum(p.region_count for p in page_logs)
        llm_invocation_rate = total_llm_invocations / max(1, total_regions)

        return ProcessingLog(
            pages=page_logs,
            total_duration_s=total_duration_s,
            total_cost_usd=total_cost_usd,
            total_llm_invocations=total_llm_invocations,
            total_hitl_flags=total_hitl_flags,
            llm_invocation_rate=llm_invocation_rate,
            engine_versions=dict(ENGINE_VERSIONS),
        )
