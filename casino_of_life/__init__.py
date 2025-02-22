"""
Casino of Life - A reinforcement learning environment for retro games
"""

from casino_of_life.game_environments.retro_env_loader import RetroEnv
from casino_of_life.utils.config import DATA_DIR
from casino_of_life.agents.custom_agent import BaseAgent
from casino_of_life.agents.dynamic_agent import DynamicAgent
from casino_of_life.agents.agent_orchestrator import AgentOrchestrator
from casino_of_life.agents.caballo_loko import CaballoLoko

__version__ = "0.2.4"

__all__ = [
    # Environment
    'RetroEnv',
    
    # Configuration
    'DATA_DIR',
    
    # Agents
    'BaseAgent',
    'DynamicAgent',
    'AgentOrchestrator',
    'CaballoLoko'
]
