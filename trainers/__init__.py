"""
Trainers - Provides various training implementations for the Casino of Life
"""

from .rl_algorithms import (
    PolicyType,
    rl_algorithm,
    create_ppo_model,
    create_a2c_model,
    create_dqn_model
)
from .brute import BruteTrainer
from .interactive import InteractiveTrainer
from .ppo2 import PPO2Trainer
from .random_agent import RandomAgentTrainer
from .retro_interactive import RetroInteractiveTrainer

__all__ = [
    # RL Algorithms
    'PolicyType',
    'rl_algorithm',
    'create_ppo_model',
    'create_a2c_model',
    'create_dqn_model',
    
    # Trainers
    'BruteTrainer',
    'InteractiveTrainer',
    'PPO2Trainer',
    'RandomAgentTrainer',
    'RetroInteractiveTrainer'
]
