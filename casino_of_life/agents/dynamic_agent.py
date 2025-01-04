# casino_of_life_/agents/dynamic_agent.py
import logging
from casino_of_life.src.client_bridge.parser import parse_user_input
from typing import Optional, Dict, Any
import json

class DynamicAgent:
    """Class for creating customizable agents based on user input."""

    def __init__(self, retro_api, rl_algorithm, training_params=None, reward_evaluators=None):
      """
          Initializes a DynamicAgent.

          Args:
            retro_api: RetroAPI dependency.
            rl_algorithm: RL algorithm to use.
            training_params: default training parameters.
            reward_evaluators: reward evaluators to use for training.
      """
      self.retro_api = retro_api
      self.rl_algorithm = rl_algorithm
      self.training_params = training_params if training_params else {}
      self.reward_evaluators = reward_evaluators if reward_evaluators else {}


    async def create_agent_from_user_input(self, user_input: str, base_agent_class):
        """
        Creates and trains an agent based on natural language input.
        
        Args:
            user_input: Natural language message from user
            base_agent_class: The class of the base agent used for training
        """
        try:
            print(f"Processing natural language input: {user_input}")
            
            # Parse the natural language input
            parsed_data = parse_user_input(user_input)
            print(f"Parsed parameters: {parsed_data}")
            
            if "error" in parsed_data:
                return {"message": f"Failed to parse input: {parsed_data['error']}"}

            return await self._train_agent(parsed_data, base_agent_class)

        except Exception as e:
            logging.error(f"Error creating agent: {e}")
            raise


    async def _train_agent(self, parsed_data: dict, base_agent_class):
        """Trains an agent with parsed parameters."""
        try:
            game = parsed_data.get("game")
            state = parsed_data.get("save_state")
            scenario = parsed_data.get("scenario")
            players = parsed_data.get("players", 1)
            total_timesteps = parsed_data.get("timesteps", 100000)
            learning_rate = parsed_data.get("learning_rate", 0.001)
            batch_size = parsed_data.get("batch_size", 64)
            checkpoint_name = f"{game}_{state}_{total_timesteps}"
            reward_evaluator = self.reward_evaluators.get(game, None)

            print(f"Creating agent with game={game}, state={state}")
            agent = base_agent_class(
                game=game,
                state=state,
                retro_api=self.retro_api,
                rl_algorithm=self.rl_algorithm,
                scenario=scenario,
                players=players,
            )

            training_params = self.training_params.copy()
            if 'learning_rate' in training_params:
                del training_params['learning_rate']
            if 'batch_size' in training_params:
                del training_params['batch_size']
            if 'timesteps' in training_params:
                del training_params['timesteps']
            if 'checkpoint_name' in training_params:
                del training_params['checkpoint_name']
            if 'reward_evaluator' in training_params:
                del training_params['reward_evaluator']

            path = await agent.train(
                total_timesteps=total_timesteps,
                learning_rate=learning_rate,
                batch_size=batch_size,
                checkpoint_name=checkpoint_name,
                reward_evaluator=reward_evaluator,
                **training_params
            )
            return {"message": f"Agent trained successfully and saved to {path}"}
        except Exception as e:
            logging.error(f"Training error: {e}")
            raise

    async def _step_agent(self, parsed_data: dict):
        """
          Steps the environment forward, and returns the result.
          Args:
           parsed_data: dictionary with the data parsed by spacy.
          Returns:
           The observation and reward for the step.
        """
        try:
            game = parsed_data.get("game")
            character = parsed_data.get("character")
            state = parsed_data.get("state")
            scenario = parsed_data.get("scenario")
            players = parsed_data.get("players")
            action = parsed_data.get("action")

            return await self.retro_api.step(action, game, state, scenario, players, character)
        except Exception as e:
            logging.error(f"Error stepping environment: {e}")
            raise

    async def _reset_agent(self, parsed_data: dict):
      """
          Resets the environment and returns the new state.
           Args:
              parsed_data: dictionary with the data parsed by spacy.
          Returns:
              The new observation from the reset.
      """
      try:
          game = parsed_data.get("game")
          state = parsed_data.get("state")
          scenario = parsed_data.get("scenario")
          players = parsed_data.get("players", 1)
          character = parsed_data.get("character")
          return await self.retro_api.reset(game, state, scenario, players, character)
      except Exception as e:
          logging.error(f"Error resetting environment: {e}")
          raise
