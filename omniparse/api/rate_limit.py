"""Rate limiting configuration for API endpoints.

Provides a shared slowapi Limiter instance with per-API-key rate limiting.
Key extraction uses the X-API-Key header, falling back to 'anonymous'
for unauthenticated requests.
"""
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from starlette.requests import Request
from starlette.responses import JSONResponse, Response


def _get_api_key(request: Request) -> str:
    """Extract API key from X-API-Key header for per-key rate limiting."""
    return request.headers.get("x-api-key", "anonymous")


limiter = Limiter(key_func=_get_api_key)


def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> Response:
    """Return 429 with Retry-After header when rate limit is exceeded."""
    retry_after = exc.detail.split("per")[0].strip() if exc.detail else "60"
    response = JSONResponse(
        {"error": f"Rate limit exceeded: {exc.detail}"},
        status_code=429,
    )
    # Extract retry-after from the limiter's view_rate_limit state
    try:
        current_limit = request.state.view_rate_limit
        if current_limit:
            window_stats = limiter._limiter.get_window_stats(
                current_limit[0], *current_limit[1]
            )
            response.headers["Retry-After"] = str(window_stats.reset_time)
    except Exception:
        # Fallback: use the limit's period as retry-after
        response.headers["Retry-After"] = "60"
    return response
