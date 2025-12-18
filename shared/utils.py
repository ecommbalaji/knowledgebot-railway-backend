"""Shared utility functions."""
import httpx
from typing import Optional, Dict, Any
import logging
from fastapi import HTTPException
from starlette.status import HTTP_503_SERVICE_UNAVAILABLE

logger = logging.getLogger(__name__)


async def make_service_request(
    method: str,
    url: str,
    data: Optional[Dict[str, Any]] = None,
    json: Optional[Dict[str, Any]] = None,
    timeout: float = 30.0
) -> Dict[str, Any]:
    """Make HTTP request to internal service."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.request(
                method=method,
                url=url,
                data=data,
                json=json,
                timeout=timeout
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
            raise
        except httpx.RequestError as e:
            logger.error(f"Request error: {e}")
            raise


def dependency_unavailable_error(dependency_name: str, reason: str | None = None) -> HTTPException:
    """Return a standardized HTTPException for missing/unavailable dependencies.

    Response JSON format:
    {
      "error": "dependency_unavailable",
      "dependency": "gemini",
      "message": "Gemini client not configured",
      "reason": "optional underlying reason"
    }
    """
    payload = {
        "error": "dependency_unavailable",
        "dependency": dependency_name,
        "message": f"{dependency_name} is not configured or unavailable",
    }
    if reason:
        payload["reason"] = str(reason)

    return HTTPException(status_code=HTTP_503_SERVICE_UNAVAILABLE, detail=payload)
