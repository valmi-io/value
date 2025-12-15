"""Internal API client for Value Control Plane backend."""

from typing import Any, Optional

import httpx


class ValueControlPlaneAPI:
    """Async HTTP client for the Value Control Plane backend."""

    def __init__(
        self,
        secret: str,
        base_url: Optional[str] = None,
        timeout: float = 10.0,
    ):
        """
        Initialize the API client.

        Args:
            secret: Agent authentication secret
            base_url: Backend API base URL
            timeout: Request timeout in seconds
        """
        self.base_url = base_url or "https://api.your-backend.com"
        self.timeout = timeout
        self._headers = {
            "X-Agent-Secret": secret,
            "Content-Type": "application/json",
        }

    async def get_agent_info(self) -> dict[str, Any]:
        """
        Fetch agent context information (organization, workspace, etc.).

        Returns:
            Dict containing agent metadata

        Raises:
            httpx.HTTPError: On API request failure
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(
                f"{self.base_url}/api/v1/agent_instance/info",
                headers=self._headers,
            )
            response.raise_for_status()
            return response.json()


class SyncValueControlPlaneAPI:
    """Synchronous HTTP client for the Value Control Plane backend."""

    def __init__(
        self,
        secret: str,
        base_url: Optional[str] = None,
        timeout: float = 10.0,
    ):
        """
        Initialize the sync API client.

        Args:
            secret: Agent authentication secret
            base_url: Backend API base URL
            timeout: Request timeout in seconds
        """
        self.base_url = base_url or "https://api.your-backend.com"
        self.timeout = timeout
        self._headers = {
            "X-Agent-Secret": secret,
            "Content-Type": "application/json",
        }

    def get_agent_info(self) -> dict[str, Any]:
        """
        Fetch agent context information (organization, workspace, etc.).

        Returns:
            Dict containing agent metadata

        Raises:
            httpx.HTTPError: On API request failure
        """
        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(
                f"{self.base_url}/api/v1/agent_instance/info",
                headers=self._headers,
            )
            response.raise_for_status()
            return response.json()
