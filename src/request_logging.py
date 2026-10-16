"""JSON request metadata for HTTP observability, without request/response bodies."""

from datetime import datetime, timezone
import json
import logging
import sys
from time import perf_counter
from uuid import uuid4

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send


logger = logging.getLogger("adeguard.requests")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)


class RequestLoggingMiddleware:
    """Emit one JSON record per HTTP request and identify normal responses.

    Per-request variables stay inside __call__ so concurrent requests do not
    share IDs or timing state. Non-HTTP traffic (including startup) passes through.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id
        started = perf_counter()
        status_code = 500
        error_type = None

        async def send_with_request_id(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                MutableHeaders(scope=message)["X-Request-ID"] = request_id
            await send(message)

        try:
            await self.app(scope, receive, send_with_request_id)
        except Exception as error:
            # Keep the original server error behavior. Do not log exception
            # messages/tracebacks here because they can contain report text.
            error_type = type(error).__name__
            raise
        finally:
            level = logging.ERROR if error_type or status_code >= 500 else (
                logging.WARNING if status_code >= 400 else logging.INFO
            )
            route = scope.get("route")
            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "level": logging.getLevelName(level),
                "event": "request_failed" if error_type else "request_completed",
                "request_id": request_id,
                "method": scope["method"],
                # Use route templates, not raw URLs, query strings, or IDs.
                "path": getattr(route, "path", "<unmatched>"),
                "status_code": status_code,
                "duration_ms": round((perf_counter() - started) * 1000, 3),
            }
            if error_type:
                record["error_type"] = error_type
            logger.log(level, json.dumps(record))
