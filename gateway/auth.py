from fastapi import HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Depends
from gateway.config import settings
from gateway import metrics

_bearer = HTTPBearer(auto_error=False)


async def require_api_key(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> str:
    token = None
    if credentials:
        token = credentials.credentials
    else:
        # Also accept x-api-key header (some clients use this)
        token = request.headers.get("x-api-key")

    if not token or token not in settings.valid_api_keys:
        metrics.AUTH_FAILURES.inc()
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token
