"""
Client Bridge - Provides interfaces for game interaction and reward evaluation
"""

from .reward_evaluators import RewardEvaluatorManager
from .action_mappers import ActionMapper

__all__ = [
    'RewardEvaluatorManager',
    'ActionMapper'
]
