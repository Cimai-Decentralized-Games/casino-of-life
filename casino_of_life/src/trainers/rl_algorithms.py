# casino-of-life-retro/src/trainers/rl_algorithms.py
import logging
from typing import Optional, Dict, Any
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.ppo import MlpPolicy as PPOMlpPolicy
from stable_baselines3.ppo import CnnPolicy as PPOCnnPolicy
from stable_baselines3.a2c import MlpPolicy as A2CMlpPolicy
from stable_baselines3.a2c import CnnPolicy as A2CCnnPolicy
from stable_baselines3.dqn import MlpPolicy as DQNMlpPolicy
from stable_baselines3.dqn import CnnPolicy as DQNCnnPolicy

class PolicyType:
    PPO = "PPO"
    MLP = "MLP"
    A2C = "A2C"
    DQN = "DQN"

def rl_algorithm(policy: str, env: Any, training_params: Optional[Dict[str, Any]] = None, **kwargs):
    """
    Creates and configures the RL algorithm based on policy type and training parameters.

    Args:
        policy: Policy type (PPO, MLP, A2C, DQN)
        env: Training environment
        training_params: Dictionary containing learning_rate, batch_size, timesteps
        **kwargs: Additional arguments for the model
    """
    try:
        # Set default training parameters if none provided
        if training_params is None:
            training_params = {
                "learning_rate": 0.001,
                "batch_size": 64,
                "timesteps": 1000000
            }

        # Configure model parameters
        model_params = {
            "env": env,
            "learning_rate": training_params.get("learning_rate", 0.001),
            "batch_size": training_params.get("batch_size", 64),
            **kwargs
        }

        # Select appropriate policy and algorithm
        if policy == PolicyType.PPO:
            return create_ppo_model("MlpPolicy", **model_params)
        elif policy == PolicyType.A2C:
            return create_a2c_model("MlpPolicy", **model_params)
        elif policy == PolicyType.DQN:
            return create_dqn_model("MlpPolicy", **model_params)
        elif policy == PolicyType.MLP:
            # Default to PPO with MlpPolicy for MLP type
            return create_ppo_model("MlpPolicy", **model_params)
        else:
            raise ValueError(f"Invalid policy type: '{policy}'")

    except Exception as e:
        logging.error(f"Error creating RL algorithm: {e}")
        raise

def create_ppo_model(policy: str, *args, **kwargs):
    """Creates a PPO model with specified policy"""
    try:
        if policy == 'MlpPolicy':
            return PPO(PPOMlpPolicy, *args, **kwargs)
        elif policy == 'CnnPolicy':
            return PPO(PPOCnnPolicy, *args, **kwargs)
        else:
            raise ValueError(f"Invalid policy for PPO: '{policy}'")
    except Exception as e:
        logging.error(f"Error creating PPO model: {e}")
        raise

def create_a2c_model(policy: str, *args, **kwargs):
    """Creates an A2C model with specified policy"""
    try:
        if policy == 'MlpPolicy':
            return A2C(A2CMlpPolicy, *args, **kwargs)
        elif policy == 'CnnPolicy':
            return A2C(A2CCnnPolicy, *args, **kwargs)
        else:
            raise ValueError(f"Invalid policy for A2C: '{policy}'")
    except Exception as e:
        logging.error(f"Error creating A2C model: {e}")
        raise

def create_dqn_model(policy: str, *args, **kwargs):
    """Creates a DQN model with specified policy"""
    try:
        if policy == 'MlpPolicy':
            return DQN(DQNMlpPolicy, *args, **kwargs)
        elif policy == 'CnnPolicy':
            return DQN(DQNCnnPolicy, *args, **kwargs)
        else:
            raise ValueError(f"Invalid policy for DQN: '{policy}'")
    except Exception as e:
        logging.error(f"Error creating DQN model: {e}")
        raise