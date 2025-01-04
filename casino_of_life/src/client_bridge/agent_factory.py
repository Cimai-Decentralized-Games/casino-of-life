# casino-of-life/src/client_bridge/agent_factory.py
import json
import logging
from .chat_client import BaseChatClient, get_agent_bridge
from casino_of_life.agents.custom_agent import BaseAgent

class AgentFactory:
    def __init__(self, api_url, dynamic_agent):
        """
        Initializes the AgentFactory.

        Args:
            api_url: Chat API URL
            dynamic_agent: Dynamic agent to use for agent creation
        """
        print(f"AgentFactory initialized with URL: {api_url}")
        self.chat_client: BaseChatClient = get_agent_bridge()
        self.dynamic_agent = dynamic_agent
        self.logger = logging.getLogger(__name__)

    async def initialize(self):
        """Async initialization method"""
        await self.chat_client.connect()
        return self

    async def create_agent(self, request_type, message, game=None, state=None, scenario=None, players=1, action=None):
        """Creates an agent based on natural language input."""
        try:
            if request_type == "train":
                return await self.chat_train_agent(message, game, state, scenario, players)
            elif request_type == "step":
                return await self.step_environment(action, game, state, scenario, players)
            elif request_type == "reset":
                return await self.reset_environment(game, state, scenario, players)
            elif request_type == "random":
                return await self.create_random_agent(game, state, scenario, players)
            else:
                raise ValueError(f"Invalid request type: '{request_type}'")
        except Exception as e:
            logging.error(f"Error creating agent: {e}")
            return {"message": str(e)}

    async def chat_train_agent(self, message, game=None, state=None, scenario=None, players=1):
        """Process natural language message through chat client and create agent."""
        try:
            print(f"Processing chat training request: {message}")
            # Pass the natural language message to the dynamic agent
            return await self.dynamic_agent.create_agent_from_user_input(
                message,
                BaseAgent
            )
        except Exception as e:
            logging.error(f"Failed to create chat agent: {e}")
            return {"message": f"Failed to create agent: {str(e)}"}

    async def step_environment(self, action, game, state, scenario, players):
        """Steps the environment forward"""
        if game is None or state is None:
            raise ValueError("Game and state must be specified for stepping the environment.")
        return await self.dynamic_agent.create_agent_from_user_input(
            f"step the game with action {action}",
            game=game,
            state=state,
            scenario=scenario,
            players=players,
            action=action
        )

    async def reset_environment(self, game, state, scenario, players):
        """Resets the environment"""
        if game is None or state is None:
            raise ValueError("Game and state must be specified for resetting the environment.")
        return await self.dynamic_agent.create_agent_from_user_input(
            "reset the game",
            game=game,
            state=state,
            scenario=scenario,
            players=players
        )

    async def create_random_agent(self, game, state, scenario, players):
        """Creates an agent that takes random actions"""
        if game is None or state is None:
            raise ValueError("Game and state must be specified for a random agent")
        return "create_random_agent: not yet implemented."

    async def close(self):
        """Cleanup method to close the chat client connection"""
        await self.chat_client.close()
