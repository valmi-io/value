"""Tests for configuration."""

from value.internal.config import SDKConfig


def test_config_with_defaults() -> None:
    """Test config with default values."""
    config = SDKConfig()
    assert config.otel_endpoint == "http://localhost:4317"
    assert config.service_name == "value-control-agent"
    assert config.enable_console_export is False


def test_config_custom_values() -> None:
    """Test config with custom values."""
    config = SDKConfig(
        otel_endpoint="http://custom:4317",
        backend_url="https://custom.api",
        service_name="my-service",
        enable_console_export=True,
    )
    assert config.otel_endpoint == "http://custom:4317"
    assert config.backend_url == "https://custom.api"
    assert config.service_name == "my-service"
    assert config.enable_console_export is True
