"""
Client Bridge Module - Handles client-side operations and integrations
"""

from casino_of_life.client_bridge.parser import parse_user_input
from casino_of_life.client_bridge.reward_evaluators import (
    RewardEvaluator,
    BasicRewardEvaluator,
    StageCompleteRewardEvaluator,
    MultiObjectiveRewardEvaluator,
    RewardEvaluatorManager
)
from casino_of_life.client_bridge.action_mappers import ActionMapper
from casino_of_life.client_bridge.chat_client import (
    BaseChatClient,
    AgentBridge,
    get_chat_client,
    get_agent_bridge
)

__all__ = [
    # Parser
    'parse_user_input',
    
    # Reward Evaluators
    'RewardEvaluator',
    'BasicRewardEvaluator',
    'StageCompleteRewardEvaluator',
    'MultiObjectiveRewardEvaluator',
    'RewardEvaluatorManager',
    
    # Action Mappers
    'ActionMapper',
    
    # Chat Clients
    'BaseChatClient',
    'AgentBridge',
    'get_chat_client',
    'get_agent_bridge'
]
