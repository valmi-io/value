"""SDK client implementations."""

from typing import Any, Optional

from opentelemetry import trace

from .internal._api import SyncValueControlPlaneAPI, ValueControlPlaneAPI
from .internal.actions import ActionContext, ActionEmitter
from .internal.config import load_config_from_env
from .internal.tracing import initialize_tracing


class AsyncValueClient:
    """Asynchronous client for the Value Control SDK."""

    def __init__(
        self,
        secret: str,
        service_name: str = "value-control-agent",
        otel_endpoint: Optional[str] = None,
        backend_url: Optional[str] = None,
        enable_console_export: bool = False,
    ):
        config = load_config_from_env()
        self.secret = secret

        self._otel_endpoint = otel_endpoint or config.otel_endpoint
        self._service_name = service_name
        self._backend_url = backend_url or config.backend_url
        self._enable_console_export = enable_console_export or config.enable_console_export

        # Setup internal API client
        self._api_client = ValueControlPlaneAPI(secret=self.secret, base_url=self._backend_url)

        # Agent context attributes
        self.organization_id = None
        self.workspace_id = None
        self.agent_name = None
        self.agent_id = None
        self.value_attributes = {}
        self._tracer = None
        self.actions_emitter = None

    @property
    def api_client(self) -> ValueControlPlaneAPI:
        return self._api_client

    @property
    def tracer(self) -> Optional[trace.Tracer]:
        return self._tracer

    def action_context(self, anonymous_id: str, user_id: Optional[str] = None, **kwargs: Any) -> Any:
        """Create an action context for sending multiple actions."""
        return ActionContext(
            emitter=self.actions_emitter,
            anonymous_id=anonymous_id,
            user_id=user_id,
            **kwargs,
        )

    def action(self) -> ActionEmitter:
        return self.actions_emitter

    async def initialize(self) -> None:
        """Initialize tracer, actions_emitter, and fetch agent context from backend."""
        agent_info = await self._api_client.get_agent_info()
        self.organization_id = agent_info.get("organization_id", "unknown")
        self.workspace_id = agent_info.get("workspace_id", "unknown")
        self.agent_name = agent_info.get("name", "unknown")
        self.agent_id = agent_info.get("id", "unknown")
        self.value_attributes = {
            "value.agent.organization_id": self.organization_id,
            "value.agent.workspace_id": self.workspace_id,
            "value.agent.name": self.agent_name,
            "value.agent.id": self.agent_id,
        }

        self._tracer = initialize_tracing(
            endpoint=self._otel_endpoint,
            service_name=self._service_name,
            console_export=self._enable_console_export,
            attributes=self.value_attributes,
        )
        self.actions_emitter = ActionEmitter(tracer=self._tracer)


class ValueClient:
    """Synchronous client for the Value Control SDK."""

    def __init__(
        self,
        secret: str,
        service_name: str = "value-control-agent",
        otel_endpoint: Optional[str] = None,
        backend_url: Optional[str] = None,
        enable_console_export: bool = False,
    ):
        config = load_config_from_env()
        self.secret = secret

        self._otel_endpoint = otel_endpoint or config.otel_endpoint
        self._service_name = service_name
        self._backend_url = backend_url or config.backend_url
        self._enable_console_export = enable_console_export or config.enable_console_export

        # Setup internal API client (sync version)
        self._api_client = SyncValueControlPlaneAPI(secret=self.secret, base_url=self._backend_url)

        # Agent context attributes
        self.organization_id = None
        self.workspace_id = None
        self.agent_name = None
        self.agent_id = None
        self.value_attributes = {}
        self._tracer = None
        self.actions_emitter = None

    @property
    def api_client(self) -> SyncValueControlPlaneAPI:
        return self._api_client

    @property
    def tracer(self) -> Optional[trace.Tracer]:
        return self._tracer

    def action_context(self, anonymous_id: str, user_id: Optional[str] = None, **kwargs: Any) -> Any:
        """Create an action context for sending multiple actions."""
        return ActionContext(
            emitter=self.actions_emitter,
            anonymous_id=anonymous_id,
            user_id=user_id,
            **kwargs,
        )

    def action(self) -> ActionEmitter:
        return self.actions_emitter

    def initialize(self) -> None:
        """Initialize tracer, actions_emitter, and fetch agent context from backend."""
        agent_info = self._api_client.get_agent_info()
        self.organization_id = agent_info.get("organization_id", "unknown")
        self.workspace_id = agent_info.get("workspace_id", "unknown")
        self.agent_name = agent_info.get("name", "unknown")
        self.agent_id = agent_info.get("id", "unknown")

        self.value_attributes = {
            "value.agent.organization_id": self.organization_id,
            "value.agent.workspace_id": self.workspace_id,
            "value.agent.name": self.agent_name,
            "value.agent.id": self.agent_id,
        }

        self._tracer = initialize_tracing(
            endpoint=self._otel_endpoint,
            service_name=self._service_name,
            console_export=self._enable_console_export,
            attributes=self.value_attributes,
        )
        self.actions_emitter = ActionEmitter(tracer=self._tracer)


def initialize_sync(
    agent_secret: str,
    service_name: str = "value-control-agent",
) -> ValueClient:
    """
    Initialize and return a configured synchronous ValueClient instance.

    Args:
        agent_secret: The agent secret for authentication (required).
        service_name: The service name for OpenTelemetry resource.

    Returns:
        A configured ValueClient instance.
    """
    client = ValueClient(secret=agent_secret, service_name=service_name)
    client.initialize()
    return client


async def initialize_async(
    agent_secret: str,
    service_name: str = "value-control-agent",
) -> AsyncValueClient:
    """
    Initialize and return a configured asynchronous AsyncValueClient instance.

    Args:
        agent_secret: The agent secret for authentication (required).
        service_name: The service name for OpenTelemetry resource.

    Returns:
        A configured AsyncValueClient instance.
    """
    client = AsyncValueClient(secret=agent_secret, service_name=service_name)
    await client.initialize()
    return client
