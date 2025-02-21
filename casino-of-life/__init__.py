"""
Casino of Life - A reinforcement learning environment for retro games
"""

# Core game environment
from .src.game_environments.retro_env_loader import RetroEnv

# Game data management
from .data import (
    GameData,
    Integrations,
    add_integrations,
    add_custom_integration,
    path,
    get_file_path,
    get_romfile_path,
    list_games,
    list_states,
    list_scenarios,
    merge,
    verify_hash,
    get_known_hashes
)

# Game wrappers and models
from .game_wrappers.models import (
    print_model_info,
    get_num_parameters,
    get_model_probabilities,
    init_model
)

# Training components
from .src.trainers import (
    PolicyType,
    rl_algorithm,
    create_ppo_model,
    create_a2c_model,
    create_dqn_model,
    BruteTrainer,
    InteractiveTrainer,
    PPO2Trainer,
    RandomAgentTrainer,
    RetroInteractiveTrainer
)
from .src.chatTrainers.chat_trainer import ChatTrainer

# Agent components
from .agents import (
    BaseAgent,
    DynamicAgent,
    AgentOrchestrator,
    CaballoLoko
)

# Client bridge components
from .src.client_bridge import (
    RewardEvaluatorManager,
    ActionMapper,
    BaseRewardEvaluator,
    BasicRewardEvaluator,
    StageCompleteRewardEvaluator,
    ProgressRewardEvaluator,
    ComboRewardEvaluator,
    MultiObjectiveRewardEvaluator,
    TournamentRewardEvaluator,
    RewardScaler,
    RewardTracker
)

__all__ = [
    # Core environment
    'RetroEnv',
    
    # Game data management
    'GameData',
    'Integrations',
    'add_integrations',
    'add_custom_integration',
    'path',
    'get_file_path',
    'get_romfile_path',
    'list_games',
    'list_states',
    'list_scenarios',
    'merge',
    'verify_hash',
    'get_known_hashes',
    
    # Game wrappers and models
    'print_model_info',
    'get_num_parameters',
    'get_model_probabilities',
    'init_model',
    
    # Training components
    'PolicyType',
    'rl_algorithm',
    'create_ppo_model',
    'create_a2c_model',
    'create_dqn_model',
    'BruteTrainer',
    'InteractiveTrainer',
    'PPO2Trainer',
    'RandomAgentTrainer',
    'RetroInteractiveTrainer',
    'ChatTrainer',
    
    # Agent components
    'BaseAgent',
    'DynamicAgent',
    'AgentOrchestrator',
    'CaballoLoko',
    
    # Client bridge components
    'RewardEvaluatorManager',
    'ActionMapper',
    'BaseRewardEvaluator',
    'BasicRewardEvaluator',
    'StageCompleteRewardEvaluator',
    'ProgressRewardEvaluator',
    'ComboRewardEvaluator',
    'MultiObjectiveRewardEvaluator',
    'TournamentRewardEvaluator',
    'RewardScaler',
    'RewardTracker'
]
