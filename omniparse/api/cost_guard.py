"""Per-page budget enforcement and cost alerting.

Provides pure functions for:
- Budget checking: abort processing when cumulative cost exceeds limit
- Threshold alerting: detect when cost crosses configurable percentage of budget
- Alert building: construct CostAlert-compatible dicts for webhook delivery
"""


class BudgetExceededError(Exception):
    """Raised when cumulative processing cost meets or exceeds the job budget.

    Attributes:
        spent: Actual cumulative cost in USD.
        budget: Configured budget limit in USD.
    """

    def __init__(self, spent: float, budget: float) -> None:
        self.spent = spent
        self.budget = budget
        super().__init__(f"Budget exceeded: ${spent:.4f} >= ${budget:.4f}")


def check_budget(cumulative_cost: float, budget_usd: float | None) -> None:
    """Check if cumulative cost exceeds the budget limit.

    No-op when budget_usd is None (unlimited budget).

    Args:
        cumulative_cost: Current total cost in USD.
        budget_usd: Budget limit in USD, or None for no enforcement.

    Raises:
        BudgetExceededError: If cumulative_cost >= budget_usd.
    """
    if budget_usd is None:
        return
    if cumulative_cost >= budget_usd:
        raise BudgetExceededError(cumulative_cost, budget_usd)


def should_alert(
    cumulative_cost: float, budget_usd: float, threshold_pct: float = 0.8
) -> bool:
    """Check if cumulative cost has crossed the alert threshold.

    Args:
        cumulative_cost: Current total cost in USD.
        budget_usd: Budget limit in USD.
        threshold_pct: Fraction of budget that triggers alert (default 0.8 = 80%).

    Returns:
        True if cumulative_cost >= budget_usd * threshold_pct.
    """
    return cumulative_cost >= budget_usd * threshold_pct


def build_cost_alert(
    job_id: str,
    api_key_hash: str,
    spent_usd: float,
    budget_usd: float,
    threshold_pct: float = 0.8,
) -> dict:
    """Build a CostAlert-compatible dict for webhook delivery.

    Args:
        job_id: The job that triggered the alert.
        api_key_hash: SHA-256 hash of the API key associated with the job.
        spent_usd: Current cumulative spend in USD.
        budget_usd: Configured budget limit in USD.
        threshold_pct: Alert threshold as fraction of budget.

    Returns:
        Dict matching CostAlert model fields.
    """
    pct_used = spent_usd / budget_usd * 100
    return {
        "job_id": job_id,
        "api_key_hash": api_key_hash,
        "spent_usd": spent_usd,
        "budget_usd": budget_usd,
        "threshold_pct": threshold_pct,
        "message": f"Job {job_id} has used {pct_used:.1f}% of budget (${spent_usd:.4f}/${budget_usd:.4f})",
    }
