# casino-of-life-retro/src/agents/agent_orchestrator.py
import logging
import asyncio
from casino_of_life.src.client_bridge.parser import parse_user_input
from casino_of_life.src.client_bridge.reward_evaluators import RewardEvaluatorManager
from casino_of_life.src.client_bridge.action_mappers import ActionMapper
from casino_of_life.src.client_bridge.chat_client import get_agent_bridge
from typing import Optional, List, Dict, Any

class AgentOrchestrator:
    """Orchestrates the interaction between Chat and game environments."""

    def __init__(self, agent_factory, game_integrations, reward_evaluators: RewardEvaluatorManager, game_controls):
        """
          Args:
            agent_factory: The agent factory for creating agents.
            game_integrations: For loading game integrations.
            reward_evaluators: For reward evaluations.
            game_controls: For the action mapping of the games.
        """
        self.agent_factory = agent_factory
        self.game_integrations = game_integrations
        self.reward_evaluators = reward_evaluators
        self.game_controls = game_controls
        
        # Map strategies to reward evaluators
        self.strategy_evaluators = {
            "aggressive": "basic",      # Focuses on damage output
            "defensive": "stage",       # Focuses on survival
            "balanced": "multi"         # Uses multiple objectives
        }

    async def handle_chat_message(self, message: Dict[str, Any]) -> dict:
        """
        Handles incoming message from Chat.
        """
        try:
            # Handle natural language messages
            if isinstance(message, str):
                parsed_message = parse_user_input(message)
                return await self._handle_natural_language_training(parsed_message)
                
            # Handle structured messages
            if "type" in message and message["type"] == "train":
                if "config" in message and "game_data" in message:
                    return await self._handle_training_request(message)
                else:
                    # Handle simple training request
                    return await self._handle_natural_language_training(message)
                    
            elif "request_type" in message:
                if message["request_type"] == "step":
                    return await self._handle_step_request(message)
                elif message["request_type"] == "reset":
                    return await self._handle_reset_request(message)
            
            logging.warning(f"Invalid request type in message: {message}")
            return {"message": "Invalid request type"}

        except Exception as e:
            logging.error(f"Error handling Chat message: {e}")
            return {"message": f"Error handling message: {e}"}

    async def _handle_training_request(self, parsed_message: Dict[str, Any]) -> dict:
        """
        Handles training requests with validation through the agent bridge.
        """
        try:
            config = parsed_message.get("config", {})
            game_data = parsed_message.get("game_data", {})

            if not config or not game_data:
                return {"message": "Missing training configuration or game data"}

            # Get agent's validation of the training configuration
            agent_bridge = get_agent_bridge()
            await agent_bridge.connect()
            
            try:
                # Validate configuration with the LLM agent
                validated_config = await agent_bridge.validate_training_params(config)
                
                # Update configuration with agent's recommendations
                strategy = validated_config.get("strategy", config.get("strategy", "balanced"))
                evaluator_type = self.strategy_evaluators.get(strategy, "multi")
                reward_evaluator = self.reward_evaluators.get_evaluator(evaluator_type)

                if not reward_evaluator:
                    return {"message": f"Invalid reward evaluator for strategy: {strategy}"}

                # Create training configuration with validated parameters
                training_config = {
                    "game": config["game"],
                    "state": config["save_state"],
                    "fighter": config["fighter"],
                    "policy": validated_config.get("policy", config["policy"]),
                    "reward_evaluator": reward_evaluator,
                    "scenario": game_data.get("data", {}).get("scenario", {}),
                    "training_params": validated_config.get("training_params", config.get("training_params", {})),
                    "players": config.get("players", 1)
                }

                # Create and train the agent
                response = await self.agent_factory.create_agent(
                    request_type="train",
                    message="Starting training session",
                    **training_config
                )

                return {
                    "success": True,
                    "message": "Training initiated successfully",
                    "config": training_config,
                    "response": response
                }

            finally:
                await agent_bridge.close()

        except Exception as e:
            logging.error(f"Error handling training request: {e}")
            return {"message": f"Error handling training request: {e}"}

    async def _handle_step_request(self, parsed_message: dict) -> dict:
        """
          Handles stepping through the game environment.

            Args:
                parsed_message: The message that was parsed, including the game state, action, etc.

            Returns:
                A dictionary containing the results from the step.
        """
        try:
            game = parsed_message.get("game")
            state = parsed_message.get("state")
            action = parsed_message.get("action")
            if game is None or state is None or action is None:
                return {"message": "The game, state, and action must be defined for stepping."}

            game_data = self.game_integrations.load_integration_data(game)
            if game_data is None:
                return {"message": f"Could not load the game: '{game}'"}

            action_mapper = ActionMapper(self.game_controls, game)
            retro_action = action_mapper.map_agent_action(action if isinstance(action, list) else [action]) # Now handles lists
            response = await self.agent_factory.create_agent(
                request_type="step",
                game=game,
                state=state,
                message = "Stepping the game",
                action = retro_action,
                scenario=parsed_message.get("scenario"),
                players=parsed_message.get("players")
            )
            obs, reward, done, info = response["observation"], response["reward"], response["done"], response["info"]

            if done:
              response = await self.agent_factory.create_agent(
                  request_type="reset",
                  game=game,
                  state=state,
                  message = "Resetting game",
                  scenario = parsed_message.get("scenario"),
                  players = parsed_message.get("players")
               )
              obs = response["observation"]
              return {"observation": obs, "reward": reward, "done": done, "info": info, "message": f"Game is done, resetted."}

            return {"observation": obs, "reward": reward, "done": done, "info": info, "message": "Stepping through game"}

        except Exception as e:
            logging.error(f"Error stepping environment: {e}")
            return {"message": f"Error stepping environment: {e}"}


    async def _handle_reset_request(self, parsed_message: dict) -> dict:
        """
         Handles resetting the game.

          Args:
            parsed_message: The message that was parsed, including the game state, action, etc.

           Returns:
               A dictionary with the new observation.
        """
        try:
            game = parsed_message.get("game")
            state = parsed_message.get("state")
            if game is None or state is None:
                return {"message": "The game, and state must be defined for resetting."}

            game_data = self.game_integrations.load_integration_data(game)
            if game_data is None:
                return {"message": f"Could not load the game: '{game}'"}

            response = await self.agent_factory.create_agent(
                request_type="reset",
                message="Resetting the game",
                game=game,
                state=state,
                scenario = parsed_message.get("scenario"),
                players = parsed_message.get("players")
            )
            obs = response["observation"]
            return {"observation": obs, "message": "Resetted game."}

        except Exception as e:
            logging.error(f"Error resetting the game: {e}")
            return {"message": f"Error resetting the game: {e}"}

    async def _handle_natural_language_training(self, parsed_message: Dict[str, Any]) -> dict:
        """
        Handles training requests from natural language input
        """
        try:
            # Get basic configuration from parsed message
            game = parsed_message.get("game", "MortalKombatII-Genesis")
            state = parsed_message.get("save_state", "Level1.LiuKangVsJax")
            strategy = parsed_message.get("strategy", "balanced")
            
            # Get appropriate reward evaluator
            evaluator_type = self.strategy_evaluators.get(strategy, "multi")
            reward_evaluator = self.reward_evaluators.get_evaluator(evaluator_type)

            if not reward_evaluator:
                return {"message": f"Invalid reward evaluator for strategy: {strategy}"}

            # Create basic training configuration
            training_config = {
                "game": game,
                "state": state,
                "scenario": None,  # No scenario for basic training
                "players": parsed_message.get("players", 1),
                "reward_evaluator": reward_evaluator
            }

            # Create and train the agent
            response = await self.agent_factory.create_agent(
                request_type="train",
                message=parsed_message,  # Pass the original parsed message
                **training_config
            )

            return {
                "success": True,
                "message": "Training initiated successfully",
                "config": training_config,
                "response": response
            }

        except Exception as e:
            logging.error(f"Error handling natural language training request: {e}")
            return {"message": f"Error handling training request: {e}"}
