"""Bounded-label HTTP metrics for the current single-process API."""

from time import perf_counter

from prometheus_client import CollectorRegistry, Counter, Histogram, CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send


class RequestMetrics:
    def __init__(self) -> None:
        # Explicit registry avoids global collectors and duplicate registration
        # when tests create multiple applications in one Python process.
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "adeguard_http_requests_total",
            "Completed HTTP requests by method, route template, and response status.",
            ["method", "path", "status_code"],
            registry=self.registry,
        )
        self.duration = Histogram(
            "adeguard_http_request_duration_seconds",
            "Elapsed downstream ASGI request time in seconds, not isolated inference time.",
            ["method", "path"],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30),
            registry=self.registry,
        )

    def response(self) -> Response:
        return Response(
            content=generate_latest(self.registry),
            headers={"Content-Type": CONTENT_TYPE_LATEST},
        )


class RequestMetricsMiddleware:
    def __init__(self, app: ASGIApp, metrics: RequestMetrics) -> None:
        self.app = app
        self.metrics = metrics

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        # Scraping the endpoint must not change the measurements it exposes.
        if scope["type"] != "http" or scope["path"].rstrip("/") == "/metrics":
            await self.app(scope, receive, send)
            return

        started = perf_counter()
        status_code = 500

        async def capture_status(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, capture_status)
        finally:
            # Bound every user-controlled label: never use raw URL paths,
            # queries, request IDs, report text, or arbitrary HTTP method names.
            method = scope["method"]
            if method not in {"GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD", "TRACE", "CONNECT"}:
                method = "OTHER"
            path = getattr(scope.get("route"), "path", "<unmatched>")
            self.metrics.requests.labels(method, path, str(status_code)).inc()
            self.metrics.duration.labels(method, path).observe(perf_counter() - started)
