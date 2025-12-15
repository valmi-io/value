"""Tests for the SDK client."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from value import AsyncValueClient, ValueClient


@pytest.mark.asyncio
async def test_async_sdk_initialization() -> None:
    """Test async SDK client initialization."""
    sdk = AsyncValueClient(secret="test-secret")
    assert sdk.secret == "test-secret"

    # Mock the API client
    sdk._api_client.get_agent_info = AsyncMock(
        return_value={
            "organization_id": "org_1",
            "workspace_id": "ws_1",
            "name": "agent_1",
            "agent_id": "agent_1",
        }
    )

    await sdk.initialize()
    assert sdk.actions_emitter is not None
    assert sdk.tracer is not None


def test_sync_sdk_initialization() -> None:
    """Test sync SDK client initialization."""
    sdk = ValueClient(secret="test-secret")
    assert sdk.secret == "test-secret"

    # Mock the API client
    sdk._api_client.get_agent_info = MagicMock(
        return_value={
            "organization_id": "org_1",
            "workspace_id": "ws_1",
            "name": "agent_1",
            "agent_id": "agent_1",
        }
    )

    sdk.initialize()
    assert sdk.actions_emitter is not None
    assert sdk.tracer is not None


def test_sdk_requires_secret() -> None:
    """Test that SDK requires a secret (positional argument)."""
    # secret is now a required positional argument, so calling without it raises TypeError
    with pytest.raises(TypeError):
        AsyncValueClient()  # type: ignore


def test_custom_endpoints() -> None:
    """Test SDK with custom endpoints."""
    sdk = AsyncValueClient(
        secret="test-secret",
        otel_endpoint="http://custom:4317",
        backend_url="https://custom-backend.com",
    )
    assert sdk._otel_endpoint == "http://custom:4317"
    assert sdk.api_client.base_url == "https://custom-backend.com"


@pytest.mark.asyncio
async def test_action_emitter() -> None:
    """Test custom action creation."""
    sdk = AsyncValueClient(secret="test-secret")

    # Mock the API client
    sdk._api_client.get_agent_info = AsyncMock(return_value={})

    await sdk.initialize()

    # Test sending action without context
    sdk.action().send(
        action_name="test_action",
        anonymous_id="anon456",
        user_id="user123",
        **{"value.action.description": "Test action"},
    )

    # Test sending action within context
    with sdk.action_context(anonymous_id="anon456", user_id="user123") as ctx:
        ctx.send(action_name="test_action_2", **{"value.action.description": "Test action 2"})


def test_anonymous_id_required() -> None:
    """Test that anonymous_id is required."""
    sdk = ValueClient(secret="test-secret")

    # Mock the API client
    sdk._api_client.get_agent_info = MagicMock(return_value={})

    sdk.initialize()

    # Test action_context requires anonymous_id
    with pytest.raises(TypeError):
        sdk.action_context(user_id="user123")

    # Test send requires anonymous_id
    with pytest.raises(TypeError):
        sdk.action().send(action_name="test_action", user_id="user123")
