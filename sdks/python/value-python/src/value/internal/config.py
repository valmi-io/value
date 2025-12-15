"""Configuration management for the SDK."""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class SDKConfig:
    """Configuration for the Value SDK."""

    otel_endpoint: str = "https://value.valmi.io"
    backend_url: Optional[str] = None
    service_name: str = "value-control-agent"
    enable_console_export: bool = False


def load_config_from_env() -> SDKConfig:
    """Load SDK configuration from environment variables."""
    return SDKConfig(
        otel_endpoint=os.getenv("VALUE_OTEL_ENDPOINT", "https://value.valmi.io"),
        backend_url=os.getenv("VALUE_BACKEND_URL"),
        service_name=os.getenv("VALUE_SERVICE_NAME", "value-control-agent"),
        enable_console_export=os.getenv("VALUE_CONSOLE_EXPORT", "false").lower() == "true",
    )


VALUE_ACTION_ATTRIBUTES = [
    "value.action.name",
    "value.action.description",
    "value.action.type",
    "value.action.status",
    "value.action.error",
    "value.action.duration",
    "value.action.start_time",
    "value.action.end_time",
    "value.action.llm.model",
    "value.action.llm.input_tokens",
    "value.action.llm.output_tokens",
    "value.action.llm.total_tokens",
    "value.action.llm.prompt",
    "value.action.llm.response",
]
