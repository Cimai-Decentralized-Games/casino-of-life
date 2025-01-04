# casino_of_life/agents/custom_agent.py
import logging
from pathlib import Path
from stable_baselines3.common.callbacks import CheckpointCallback
from typing import Optional, Dict, Any
import gymnasium as gym
import traceback

class BaseAgent:
    """Base class for creating customizable agents."""

    def __init__(self,
                game: str,
                state: str,
                retro_api,
                rl_algorithm,
                save_dir: str = "trained_models",
                scenario: Optional[str] = None,
                players: int = 1,
                policy_kwargs: Optional[Dict[str, Any]] = None
                ):
        """
            Initialize an agent
        """
        print(f"Initializing BaseAgent with game={game}, state={state}")
        
        self.game = game
        self.state = state
        self.scenario = scenario
        self.players = players
        self.retro_api = retro_api
        self.rl_algorithm = rl_algorithm
        self.save_dir = Path(save_dir) / f"{game}_{state}"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.policy_kwargs = policy_kwargs if policy_kwargs else {'net_arch': dict(pi=[256, 256], vf=[256, 256])}

        self.model = None
        print("=== Debug Trace ===")
        print(f"Creating environment for game: {game}")
        try:
            self.env = self.retro_api.make_env(
                game=self.game,
                state=self.state,
                scenario=self.scenario,
                players=self.players
            )
        except Exception as e:
            print("Stack trace:")
            traceback.print_exc()
            raise

        self._setup_logging()


    def _setup_logging(self):
        """Sets up basic logging for the agent."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )


    def _create_checkpoint_callback(self, save_frequency: int, checkpoint_name:str="checkpoint"):
        """Creates checkpoint callback for model saving."""
        return CheckpointCallback(
            save_freq=save_frequency,
            save_path=str(self.save_dir),
            name_prefix=f"{checkpoint_name}"
        )

    def train(self,
              total_timesteps: int = 100000,
              learning_rate: float = 3e-4,
              batch_size: int = 32,
              save_frequency: int = 10000,
              checkpoint_name: str ="checkpoint",
              reward_evaluator: str = "default",
              **training_params):
        """
            Trains the agent using the given RL algorithm.
        """
        try:
            # Use policy from training_params if provided, otherwise use PPO
            policy = training_params.pop('policy', 'PPO')
            
            self.model = self.rl_algorithm(
                policy,          # Use the policy from parameters
                self.env,
                verbose=1,
                learning_rate=learning_rate,
                batch_size=batch_size,
                policy_kwargs=self.policy_kwargs,
                tensorboard_log=str(self.save_dir / 'logs'),
                **training_params
            )

            checkpoint_callback = self._create_checkpoint_callback(save_frequency, checkpoint_name)

            logging.info(f"Starting training for game: '{self.game}', at state: '{self.state}' - {total_timesteps} steps")
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=checkpoint_callback,
            )

            final_path = self.save_dir / f"{self.game}_{self.state}_final.zip"
            self.model.save(final_path)
            logging.info(f"Training complete - Model saved to {final_path}")
            return str(final_path)
        except Exception as e:
            logging.error(f"Training error: {str(e)}")
            raise

    def evaluate(self, model_path: Optional[str] = None, episodes: int = 10):
        """
        Evaluates the agent.

        Args:
          model_path: path to model if loading from a path.
          episodes: num of episodes for evaluation.

          Returns:
            dict: A dictionary with average reward, win rate, and total episodes.
        """
        if model_path:
            self.model = self.rl_algorithm.load(model_path, env=self.env)
        elif not self.model:
            raise ValueError("No model available - please train or load one first")

        try:
            total_reward = 0
            wins = 0

            for episode in range(episodes):
                obs, _ = self.env.reset()
                episode_reward = 0
                terminated = False
                truncated = False

                while not (terminated or truncated):
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = self.env.step(action)
                    episode_reward += reward

                total_reward += episode_reward
                if episode_reward > 0:
                    wins += 1

                logging.info(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")

            avg_reward = total_reward / episodes
            win_rate = wins / episodes

            return {
                'average_reward': avg_reward,
                'win_rate': win_rate,
                'total_episodes': episodes
            }
        except Exception as e:
            logging.error(f"Evaluation error: {str(e)}")
            raise
