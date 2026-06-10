"""Authentication & multi-tenancy.

Resolves the calling tenant from either an ``X-API-Key`` header (mapped to a
tenant via ``AUTH_API_KEYS``) or a Bearer JWT carrying a ``tenant`` claim. When
``ENABLE_AUTH`` is false the resolver returns the default tenant, so the platform
is single-tenant and open in local/dev mode and multi-tenant + authenticated in
production with no route changes.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Annotated

from fastapi import Depends, Header, HTTPException, status

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def create_access_token(tenant: str, expires_minutes: int = 60) -> str:
    """Mint a short-lived HS256 JWT for a tenant (handy for demos and tests)."""
    import jwt

    payload = {
        "tenant": tenant,
        "iat": datetime.now(timezone.utc),
        "exp": datetime.now(timezone.utc) + timedelta(minutes=expires_minutes),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def get_tenant(
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
    authorization: Annotated[str | None, Header()] = None,
) -> str:
    """FastAPI dependency that returns the resolved tenant id."""
    if not settings.enable_auth:
        return settings.default_tenant

    if x_api_key:
        tenant = settings.api_key_map.get(x_api_key)
        if tenant:
            return tenant
        raise _unauthorized("Invalid API key")

    if authorization and authorization.lower().startswith("bearer "):
        import jwt

        token = authorization.split(" ", 1)[1]
        try:
            payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
        except Exception as exc:  # noqa: BLE001
            raise _unauthorized(f"Invalid token: {exc.__class__.__name__}")
        tenant = payload.get("tenant") or payload.get("tid")
        if tenant:
            return tenant
        raise _unauthorized("Token missing tenant claim")

    raise _unauthorized("Authentication required")


CurrentTenant = Annotated[str, Depends(get_tenant)]
