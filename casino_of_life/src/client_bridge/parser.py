# casino-of-life/client_bridge/parser.py
import json
import logging
from typing import Dict, Any, Optional, List
from enum import Enum
from dataclasses import dataclass

class TrainingPolicy(Enum):
    PPO = "PPO"
    MLP = "MLP"
    A2C = "A2C"
    DQN = "DQN"

class Strategy(Enum):
    AGGRESSIVE = "aggressive"
    DEFENSIVE = "defensive"
    BALANCED = "balanced"

@dataclass
class ScenarioConfig:
    actions: List[List[List[str]]]
    crop: List[int]
    done: Dict[str, Any]
    reward: Dict[str, Any]

def parse_user_input(user_message: str) -> Dict[str, Any]:
    """
    Parses natural language input into training parameters
    
    Args:
        user_message: Natural language message from user (e.g., "I want Liu Kang to kick ass")
        
    Returns:
        Dictionary containing parsed training parameters
    """
    try:
        message = user_message.lower()
        parsed = {
            "game": "MortalKombatII-Genesis",
            "character": None,
            "save_state": "Level1.LiuKangVsJax",  # Default state
            "policy": "PPO",
            "strategy": "aggressive",  # Default to aggressive for "kick ass" type messages
            "timesteps": 100000,
            "players": 1,
            "learning_rate": 0.001,
            "batch_size": 64
        }

        # Basic natural language parsing
        if "liu kang" in message or "liu" in message:
            parsed["character"] = "liu_kang"
            parsed["save_state"] = "Level1.LiuKangVsJax"
        
        # Detect aggressive intent
        if any(word in message for word in ["kick ass", "destroy", "aggressive", "beat", "win"]):
            parsed["strategy"] = "aggressive"
        elif any(word in message for word in ["defend", "defensive", "survive"]):
            parsed["strategy"] = "defensive"
        else:
            parsed["strategy"] = "balanced"

        print(f"Parsed message '{message}' into parameters: {parsed}")
        return parsed

    except Exception as e:
        logging.error(f"Failed to parse user input: {e}")
        return {"error": str(e)}

def modify_scenario_rewards(scenario: ScenarioConfig, strategy: Strategy) -> ScenarioConfig:
    """Modifies scenario rewards based on strategy"""
    if strategy == Strategy.AGGRESSIVE:
        scenario.reward["variables"]["enemy_health"]["penalty"] = -2.0
        scenario.reward["variables"]["health"]["penalty"] = 0.5
    elif strategy == Strategy.DEFENSIVE:
        scenario.reward["variables"]["enemy_health"]["penalty"] = -0.5
        scenario.reward["variables"]["health"]["penalty"] = 2.0
    return scenario
