# casino-of-life/src/client_bridge/chat_client.py
from abc import ABC, abstractmethod
import json
import httpx
from typing import Dict, Any, Optional
from casino_of_life.src.utils.config import CHAT_WS_URL, CHAT_API_KEY

class BaseChatClient(ABC):
    """Abstract base class for interfacing with casino-of-life"""
    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the service"""
        pass

    @abstractmethod
    async def get_training_config(self, message: str) -> Dict[str, Any]:
        """Get training configuration"""
        pass

    @abstractmethod
    async def validate_training_params(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate training parameters"""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the connection"""
        pass

class AgentBridgeClient(BaseChatClient):
    """Bridge implementation to communicate with casino-of-life"""
    def __init__(self, agent_url: str = None, api_key: Optional[str] = None):
        print(f"AgentBridgeClient.__init__ received URL: {agent_url}")  # Debug print
        # Use the config values if no override is provided
        self.agent_url = agent_url or CHAT_WS_URL
        print(f"AgentBridgeClient using URL: {self.agent_url}")  # Debug print
        self.api_key = api_key or CHAT_API_KEY
        self.client = None

    async def connect(self) -> None:
        """Initialize connection to service"""
        print(f"Connecting to URL: {self.agent_url}")
        self.client = httpx.AsyncClient(
            base_url=self.agent_url,
            headers={'Authorization': f'Bearer {self.api_key}'} if self.api_key else {}
        )
        print("Connected to casino-of-life service")

    async def get_training_config(self, message: str) -> Dict[str, Any]:
        """Get training configuration"""
        if not self.client:
            raise ConnectionError("Client not initialized. Call connect() first.")

        try:
            payload = {
                "type": "train",
                "message": message,
                "game": "MortalKombatII-Genesis",
                "state": "Level1.LiuKangVsJax",
                "scenario": None,
                "players": 1
            }
            response = await self.client.post("/train", json=payload)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            raise ConnectionError(f"Failed to get training config: {e}") from e

    async def validate_training_params(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate training parameters"""
        if not self.client:
            raise ConnectionError("Client not initialized. Call connect() first.")

        try:
            response = await self.client.post("/train/validate", json=config)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            raise ConnectionError(f"Failed to validate training params: {e}") from e

    async def close(self) -> None:
        """Close connection"""
        if self.client:
            await self.client.aclose()
            print("Disconnected from casino-of-life service")

def get_agent_bridge() -> BaseChatClient:
    """Returns the bridge client for casino-of-life communication"""
    print(f"get_agent_bridge using CHAT_WS_URL: {CHAT_WS_URL}")  # Debug print
    return AgentBridgeClient(agent_url=CHAT_WS_URL)

def get_chat_client() -> BaseChatClient:
    """Maintains backward compatibility"""
    return get_agent_bridge()
