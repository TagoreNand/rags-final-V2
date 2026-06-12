"""
backend/auth/middleware.py
API-key and JWT bearer authentication for FastAPI.

Usage (in app.py):
    from backend.auth.middleware import AuthMiddleware, require_auth
    app.add_middleware(AuthMiddleware)   # attaches to every route
    # OR use the dependency on individual routes:
    @router.post("/v1/tasks", dependencies=[Depends(require_auth)])
"""

from __future__ import annotations

import time
from typing import Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from backend.config import get_settings
from backend.tracing import get_logger

try:
    import jwt as pyjwt

    _JWT_AVAILABLE = True
except ImportError:
    _JWT_AVAILABLE = False

log = get_logger(__name__)
_bearer = HTTPBearer(auto_error=False)


# ── Token helpers ─────────────────────────────────────────────────────────────


def create_access_token(sub: str) -> str:
    """Create a signed JWT for `sub`. Useful for /v1/auth/token endpoint."""
    cfg = get_settings()
    if not _JWT_AVAILABLE:
        raise RuntimeError("PyJWT not installed; cannot issue tokens.")
    payload = {
        "sub": sub,
        "iat": int(time.time()),
        "exp": int(time.time()) + cfg.jwt_expire_minutes * 60,
    }
    return pyjwt.encode(payload, cfg.jwt_secret, algorithm=cfg.jwt_algorithm)


def _verify_jwt(token: str) -> Optional[str]:
    """Returns `sub` claim or None on any failure."""
    cfg = get_settings()
    if not _JWT_AVAILABLE:
        return None
    try:
        data = pyjwt.decode(token, cfg.jwt_secret, algorithms=[cfg.jwt_algorithm])
        return data.get("sub")
    except Exception:
        return None


# ── FastAPI dependency ────────────────────────────────────────────────────────


async def require_auth(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> str:
    """
    FastAPI dependency.  Returns the authenticated identity (api-key or JWT sub).
    Raises 401 if auth is enabled and credentials are invalid.
    Passes through (returns "anonymous") if auth is disabled.
    """
    cfg = get_settings()
    if not cfg.auth_enabled:
        return "anonymous"

    # 1. Check X-API-Key header first (simpler for scripts)
    api_key_header = request.headers.get("X-API-Key", "")
    if api_key_header and api_key_header in cfg.api_keys:
        log.debug("auth_ok", method="api_key")
        return api_key_header

    # 2. Bearer JWT
    if credentials and credentials.scheme.lower() == "bearer":
        sub = _verify_jwt(credentials.credentials)
        if sub:
            log.debug("auth_ok", method="jwt", sub=sub)
            return sub

    log.warning("auth_failed", path=str(request.url))
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing credentials.",
        headers={"WWW-Authenticate": "Bearer"},
    )
