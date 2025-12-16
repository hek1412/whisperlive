"""
Rate limiting for WhisperLive API endpoints.
Uses slowapi with Redis backend for distributed rate limiting.
"""

import logging
from typing import Optional
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request

logger = logging.getLogger(__name__)


def get_api_key_or_ip(request: Request) -> str:
    """
    Get rate limit key based on API key or IP address.

    Args:
        request: FastAPI request object

    Returns:
        Rate limit key (API key or IP address)
    """
    # Try to get API key from query params
    api_key = request.query_params.get("api_key")
    if api_key:
        return f"api_key:{api_key}"

    # Try to get API key from headers
    api_key = request.headers.get("x-api-key")
    if api_key:
        return f"api_key:{api_key}"

    # Fall back to IP address
    ip_address = get_remote_address(request)
    return f"ip:{ip_address}"


def create_rate_limiter(
    redis_url: Optional[str] = None,
    enabled: bool = True
) -> Optional[Limiter]:
    """
    Create rate limiter instance.

    Args:
        redis_url: Redis connection URL for distributed rate limiting
                   If None, uses in-memory storage (not recommended for production)
        enabled: If False, rate limiting is disabled

    Returns:
        Limiter instance or None if disabled
    """
    if not enabled:
        logger.info("[RATE_LIMITER] Rate limiting disabled")
        return None

    try:
        if redis_url:
            logger.info(f"[RATE_LIMITER] Initializing with Redis backend: {redis_url}")
            limiter = Limiter(
                key_func=get_api_key_or_ip,
                storage_uri=redis_url,
                strategy="fixed-window"
            )
        else:
            logger.warning("[RATE_LIMITER] Using in-memory storage (not recommended for production)")
            limiter = Limiter(
                key_func=get_api_key_or_ip,
                strategy="fixed-window"
            )

        logger.info("[RATE_LIMITER] Rate limiter initialized successfully")
        return limiter

    except Exception as e:
        logger.error(f"[RATE_LIMITER] Failed to initialize rate limiter: {e}")
        return None


# Default rate limits for different endpoint types
RATE_LIMITS = {
    # Session management endpoints
    "create_session": "10/minute",     # 10 session creations per minute
    "get_session": "60/minute",        # 60 session queries per minute
    "close_session": "20/minute",      # 20 session closures per minute
    "delete_session": "20/minute",     # 20 session deletions per minute

    # Transcript endpoints
    "get_transcript": "60/minute",     # 60 transcript retrievals per minute
    "get_summary": "30/minute",        # 30 summary retrievals per minute

    # WebSocket (per connection)
    "websocket_connect": "5/minute",   # 5 WebSocket connections per minute

    # Health and status
    "health_check": "100/minute",      # 100 health checks per minute

    # Admin endpoints
    "list_sessions": "30/minute",      # 30 list operations per minute
}


def get_rate_limit(endpoint: str) -> str:
    """
    Get rate limit string for an endpoint.

    Args:
        endpoint: Endpoint name

    Returns:
        Rate limit string (e.g., "10/minute")
    """
    return RATE_LIMITS.get(endpoint, "60/minute")


def setup_rate_limiter(app, redis_url: Optional[str] = None, enabled: bool = True):
    """
    Setup rate limiter for FastAPI app.

    Args:
        app: FastAPI application instance
        redis_url: Redis connection URL
        enabled: If False, rate limiting is disabled
    """
    if not enabled:
        logger.info("[RATE_LIMITER] Rate limiting is disabled")
        app.state.limiter = None
        return

    limiter = create_rate_limiter(redis_url=redis_url, enabled=enabled)

    if limiter:
        app.state.limiter = limiter
        app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
        logger.info("[RATE_LIMITER] Rate limiter configured for app")
    else:
        app.state.limiter = None
        logger.warning("[RATE_LIMITER] Rate limiter not configured")


class RateLimitConfig:
    """Configuration for rate limiting."""

    def __init__(
        self,
        redis_url: Optional[str] = None,
        enabled: bool = True,
        custom_limits: Optional[dict] = None
    ):
        """
        Initialize rate limit configuration.

        Args:
            redis_url: Redis connection URL
            enabled: Enable/disable rate limiting
            custom_limits: Custom rate limits dictionary (overrides defaults)
        """
        self.redis_url = redis_url
        self.enabled = enabled

        # Merge custom limits with defaults
        self.limits = RATE_LIMITS.copy()
        if custom_limits:
            self.limits.update(custom_limits)

    def get_limit(self, endpoint: str) -> str:
        """
        Get rate limit for an endpoint.

        Args:
            endpoint: Endpoint name

        Returns:
            Rate limit string
        """
        return self.limits.get(endpoint, "60/minute")

    def __repr__(self) -> str:
        return (
            f"RateLimitConfig(enabled={self.enabled}, "
            f"redis_url={self.redis_url}, "
            f"endpoints={len(self.limits)})"
        )
