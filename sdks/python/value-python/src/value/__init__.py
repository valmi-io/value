"""Value Python SDK for OpenTelemetry-based AI agent observability and control."""

from .client import AsyncValueClient, ValueClient, initialize_async, initialize_sync
from .instrumentation import (
    auto_instrument,
    get_supported_libraries,
    is_library_available,
    uninstrument,
)

__version__ = "0.1.4"
__all__ = [
    "ValueClient",
    "AsyncValueClient",
    "auto_instrument",
    "uninstrument",
    "get_supported_libraries",
    "is_library_available",
    "initialize_async",
    "initialize_sync",
]
