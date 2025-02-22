# casino_of_life/src/chatTrainers/chat_trainer.py
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from client_bridge.chat_client import ChatClient

class ChatTrainer:
    """
        Enhanced Trainer with Chat Conversational Interface
        Provides an intuitive, conversational approach to game AI training.
    """
    def __init__(self,
                 api_url,
                 retro_api,
                 rl_algorithm,
                 save_dir: str = 'trained_models',
                 ):
        """
        Initializes the Chat Trainer

        Args:
            api_url: URL to the Chat API server.
            retro_api: RetroAPI dependency for interacting with game environments.
            rl_algorithm: RL algorithm to use for training.
            save_dir: Directory to save trained models.
        """
        self.chat_client = ChatClient(api_url)
        self.retro_api = retro_api
        self.rl_algorithm = rl_algorithm
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.training_config = {}

        self._setup_logging()

    def _setup_logging(self):
        """ Configure logging for the trainer. """
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )


    async def configure_training(self, agent, config: Dict[str, Any] = None):
        """
          Configures training parameters through the Chat conversational interface.

          Args:
              agent: The agent that will be used for training.
              config: Dictionary with configuration parameters.
         """
        if config:
            self.training_config.update(config)
        else:
          await self._interactive_configuration(agent)

    async def _interactive_configuration(self, agent):
       """ Interactive configuration through Chat """
       # Get the agent params
       if not agent:
            return "You need to define what character to train first."
       # Use Chat to get the correct training parameters.
       response = await self.chat_client.send_message(
           f"What are the training configurations for the agent: {agent}"
       )

       if "training_params" in response:
            self.training_config.update(response["training_params"])
       else:
          logging.warning(f"Could not get training parameters from Chat: {response}")
          return "Could not get training parameters from Chat, use default."
       return response


    async def train_agent(self, base_agent_class, agent_params = None):
        """
          Trains the agent with configuration parameters.

          Args:
             base_agent_class: The base agent that will be used for the training.
            agent_params: Training parameters, such as the game, state, etc.
        """
        try:
            if agent_params is None:
                return "You need to define agent training parameters to begin."

            if not "game" in agent_params or not "state" in agent_params:
              return "You need to define the 'game', and the 'state'."

            agent = base_agent_class(
                game = agent_params["game"],
                state = agent_params["state"],
                retro_api = self.retro_api,
                rl_algorithm = self.rl_algorithm,
                save_dir = str(self.save_dir),
                scenario=agent_params.get("scenario", None),
                players = agent_params.get("players", 1),
            )
            path = agent.train(**self.training_config)
            return path
        except Exception as e:
            logging.error(f"Training failed: {str(e)}")
            raise