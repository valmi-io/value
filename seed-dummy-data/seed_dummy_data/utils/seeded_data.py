"""Seeded data manager for persisting and loading agent data."""

import json
import os
import random
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

logger = structlog.get_logger(__name__)

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
DEFAULT_DATA_FILE = os.path.join(_DATA_DIR, "seeded_agents.json")


class AnonymousIdSelector:
    def __init__(self, anonymous_ids: List[str]):
        self.ids = anonymous_ids
        self.index = 0

    def get_next_batch(self) -> List[str]:
        if not self.ids:
            return []
        count = random.randint(1, 3)
        batch = []
        for _ in range(count):
            batch.append(self.ids[self.index % len(self.ids)])
            self.index += 1
        return batch

    def reset(self):
        self.index = 0


class SeededDataManager:
    def __init__(self, filepath: Optional[str] = None):
        self.filepath = Path(filepath or DEFAULT_DATA_FILE)
        self.data: Dict[str, Any] = {}
        self._selectors: Dict[str, AnonymousIdSelector] = {}

    def exists(self) -> bool:
        return self.filepath.exists()

    def load(self) -> bool:
        if not self.exists():
            logger.warning("Seeded data file not found", filepath=str(self.filepath))
            return False
        try:
            with open(self.filepath, "r") as f:
                self.data = json.load(f)
            logger.info(
                "Loaded seeded data", filepath=str(self.filepath), agents_count=len(self.data.get("agents", []))
            )
            self._init_selectors()
            return True
        except json.JSONDecodeError as e:
            logger.error("Failed to parse seeded data file", error=str(e))
            return False
        except Exception as e:
            logger.error("Failed to load seeded data", error=str(e))
            return False

    def save(self) -> bool:
        try:
            with open(self.filepath, "w") as f:
                json.dump(self.data, f, indent=2, default=str)
            logger.info("Saved seeded data", filepath=str(self.filepath), agents_count=len(self.data.get("agents", [])))
            return True
        except Exception as e:
            logger.error("Failed to save seeded data", error=str(e))
            return False

    def _init_selectors(self):
        self._selectors = {}
        for agent in self.data.get("agents", []):
            agent_id = agent.get("agent_id")
            anonymous_ids = agent.get("anonymous_ids", [])
            if agent_id and anonymous_ids:
                self._selectors[agent_id] = AnonymousIdSelector(anonymous_ids)

    def set_data(self, workspace_id: str, organization_id: str, agents: List[Dict[str, Any]]):
        self.data = {
            "workspace_id": workspace_id,
            "organization_id": organization_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "agents": agents,
        }
        self._init_selectors()

    def get_workspace_id(self) -> Optional[str]:
        return self.data.get("workspace_id")

    def get_organization_id(self) -> Optional[str]:
        return self.data.get("organization_id")

    def get_agents(self) -> List[Dict[str, Any]]:
        return self.data.get("agents", [])

    def get_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        for agent in self.get_agents():
            if agent.get("agent_id") == agent_id:
                return agent
        return None

    def get_anonymous_ids(self, agent_id: str) -> List[str]:
        agent = self.get_agent(agent_id)
        if agent:
            return agent.get("anonymous_ids", [])
        return []

    def get_next_anonymous_ids(self, agent_id: str) -> List[str]:
        selector = self._selectors.get(agent_id)
        if selector:
            return selector.get_next_batch()
        return []

    def reset_selectors(self):
        for selector in self._selectors.values():
            selector.reset()

    @staticmethod
    def generate_anonymous_ids(min_count: int = 1, max_count: int = 10) -> List[str]:
        count = random.randint(min_count, max_count)
        return [str(uuid.uuid4()) for _ in range(count)]
