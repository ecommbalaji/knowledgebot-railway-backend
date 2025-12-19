"""Shared utility functions."""
import httpx
from typing import Optional, Dict, Any
import logging
from fastapi import HTTPException, FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.status import HTTP_503_SERVICE_UNAVAILABLE
import sys
import asyncio
import signal
import traceback
import faulthandler
from tenacity import retry, stop_after_attempt, wait_exponential
import psutil

logger = logging.getLogger(__name__)


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
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


def setup_global_exception_logging(service_name: str) -> None:
    """Install global exception and signal handlers that log full tracebacks.

    This configures:
    - sys.excepthook to log uncaught exceptions
    - asyncio event loop exception handler to capture async errors
    - SIGTERM / SIGINT handlers that dump python tracebacks for diagnostics
    """
    svc_logger = logging.getLogger(service_name)

    def _excepthook(exc_type, exc_value, exc_tb):
        # Log an uncaught exception with full traceback
        try:
            svc_logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_tb))
        except Exception:
            # Fallback to printing if logging fails
            traceback.print_exception(exc_type, exc_value, exc_tb)

    sys.excepthook = _excepthook

    # Asyncio exception handler
    try:
        loop = asyncio.get_event_loop()

        def _async_exc_handler(loop, context):
            try:
                msg = context.get("message", "Unhandled async exception")
                svc_logger.error(f"Unhandled async exception: {msg}")
                if "exception" in context and context["exception"] is not None:
                    svc_logger.error("Async exception detail:", exc_info=context["exception"])
                else:
                    svc_logger.error(f"Async context: {context}")
            except Exception:
                svc_logger.exception("Failed while logging asyncio exception")

        loop.set_exception_handler(_async_exc_handler)
    except RuntimeError:
        # Event loop may not exist yet
        svc_logger.debug("No running asyncio event loop to set exception handler on")

    # Signal handlers to capture shutdown requests
    def _signal_handler(signum, frame):
        try:
            svc_logger.warning(f"Received signal {signum} - initiating shutdown; dumping tracebacks for diagnostics")
            # Dump traceback of all threads to stderr (faulthandler is robust)
            try:
                faulthandler.dump_traceback(file=sys.stderr)
            except Exception:
                svc_logger.exception("faulthandler failed to dump traceback")
        except Exception:
            svc_logger.exception("Error inside signal handler")

    for s in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(s, _signal_handler)
        except Exception:
            # Some environments (Windows, restricted containers) may not allow setting signals
            svc_logger.debug(f"Could not register signal handler for {s}")


def register_fastapi_exception_handlers(app: FastAPI, service_name: str) -> None:
    """Register a FastAPI exception handler that logs full traceback and returns a safe JSON response.

    Use this to ensure request-scoped exceptions are logged with stack traces.
    """
    svc_logger = logging.getLogger(service_name)

    @app.exception_handler(Exception)
    async def _global_exc_handler(request: Request, exc: Exception):
        svc_logger.error("Unhandled exception during request processing", exc_info=True)
        # Return a minimal safe JSON payload to callers
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_server_error",
                "message": "An unexpected error occurred. Check service logs for details."
            },
        )


def log_system_metrics(service_name: str) -> None:
    """Log system memory and CPU usage metrics for a given service.

    Args:
        service_name: Name of the service for logging context.
    """
    svc_logger = logging.getLogger(service_name)
    try:
        memory = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=1)
        svc_logger.info(f"📊 System Metrics for {service_name}:")
        svc_logger.info(f"  Memory: {memory.used / (1024 ** 2):.2f} MB / {memory.total / (1024 ** 2):.2f} MB ({memory.percent}%)")
        svc_logger.info(f"  CPU Usage: {cpu}%")
    except Exception as e:
        svc_logger.error(f"Error logging system metrics: {e}")
