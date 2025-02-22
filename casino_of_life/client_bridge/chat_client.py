"""
Chat Client Module - Handles chat-based interactions and agent communication
"""

import logging
import asyncio
from typing import Dict, Any, Optional
import aiohttp

class BaseChatClient:
    """Base class for chat-based interactions"""
    
    def __init__(self, api_url: str = None):
        self.api_url = api_url or "ws://localhost:6789/ws"
        self.websocket = None
        self.connected = False

    async def connect(self):
        """Establish WebSocket connection"""
        try:
            if not self.connected:
                session = aiohttp.ClientSession()
                self.websocket = await session.ws_connect(self.api_url)
                self.connected = True
                logging.info("Connected to chat server")
        except Exception as e:
            logging.error(f"Failed to connect to chat server: {e}")
            raise

    async def close(self):
        """Close WebSocket connection"""
        if self.websocket:
            await self.websocket.close()
            self.connected = False
            logging.info("Disconnected from chat server")

    async def send_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Send message to chat server"""
        try:
            if not self.connected:
                await self.connect()
            await self.websocket.send_json(message)
            response = await self.websocket.receive_json()
            return response
        except Exception as e:
            logging.error(f"Error sending message: {e}")
            raise

class AgentBridge(BaseChatClient):
    """Bridge for agent-specific chat interactions"""
    
    async def validate_training_params(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and potentially modify training parameters
        
        Args:
            config: Training configuration to validate
            
        Returns:
            Validated and potentially modified configuration
        """
        try:
            message = {
                "type": "validate_training",
                "config": config
            }
            response = await self.send_message(message)
            return response.get("config", config)
        except Exception as e:
            logging.error(f"Error validating training parameters: {e}")
            return config

def get_chat_client(api_url: Optional[str] = None) -> BaseChatClient:
    """Get a chat client instance"""
    return BaseChatClient(api_url)

def get_agent_bridge(api_url: Optional[str] = None) -> AgentBridge:
    """Get an agent bridge instance"""
    return AgentBridge(api_url)
