"""
backend/tracing/setup.py
Structured logging (structlog) + OpenTelemetry tracing bootstrapper.

Call `configure_telemetry()` once at app startup.
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Dict, Generator, Optional

import structlog

# ── Optional OTEL imports (graceful degradation if not installed) ─────────────
try:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False


_tracer: Optional[Any] = None  # opentelemetry Tracer or None


def configure_telemetry(
    service_name: str,
    otlp_endpoint: str,
    enabled: bool,
    log_level: str,
    log_format: str,
) -> None:
    """Idempotent – safe to call multiple times."""
    _configure_logging(log_level, log_format)
    if enabled and _OTEL_AVAILABLE:
        _configure_otel(service_name, otlp_endpoint)


# ── Logging ──────────────────────────────────────────────────────────────────


def _configure_logging(level: str, fmt: str) -> None:
    # Note: structlog.stdlib.add_logger_name requires a stdlib Logger backend;
    # we use PrintLoggerFactory so we omit it and embed the name in get_logger() calls.
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]
    if fmt == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(level.upper())
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(sys.stdout),
        cache_logger_on_first_use=False,  # allow reconfigure in tests
    )

    # Also configure stdlib logging so third-party libs are captured.
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.getLevelName(level.upper()),
    )


def get_logger(name: str = __name__) -> Any:
    return structlog.get_logger(name)


# ── OpenTelemetry ─────────────────────────────────────────────────────────────


def _configure_otel(service_name: str, otlp_endpoint: str) -> None:
    global _tracer
    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer(service_name)


@contextmanager
def span(
    name: str, attributes: Optional[Dict[str, Any]] = None
) -> Generator[Any, None, None]:
    """Context manager that creates an OTEL span if available, else is a no-op."""
    if _OTEL_AVAILABLE and _tracer is not None:
        with _tracer.start_as_current_span(name) as s:
            if attributes:
                for k, v in attributes.items():
                    s.set_attribute(k, str(v))
            yield s
    else:
        yield None


# ── Timing decorator ──────────────────────────────────────────────────────────


def timed(fn: Callable) -> Callable:
    """Logs duration and creates a span for any function."""
    log = get_logger(fn.__module__)

    @wraps(fn)
    def wrapper(*args, **kwargs):  # noqa: ANN001
        t0 = time.perf_counter()
        span_name = f"{fn.__module__}.{fn.__qualname__}"
        with span(span_name):
            try:
                result = fn(*args, **kwargs)
                elapsed = time.perf_counter() - t0
                log.debug("fn_ok", fn=span_name, elapsed_ms=round(elapsed * 1000, 1))
                return result
            except Exception as exc:
                elapsed = time.perf_counter() - t0
                log.warning(
                    "fn_err",
                    fn=span_name,
                    elapsed_ms=round(elapsed * 1000, 1),
                    error=str(exc),
                )
                raise

    return wrapper
