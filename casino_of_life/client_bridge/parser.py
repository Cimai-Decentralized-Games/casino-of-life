"""
Parser Module - Handles parsing of user input and commands
"""

import logging
from typing import Dict, Any

def parse_user_input(user_input: str) -> Dict[str, Any]:
    """
    Parse natural language input into structured parameters.
    
    Args:
        user_input: Natural language string from user
        
    Returns:
        Dictionary containing parsed parameters
    """
    try:
        # Basic parsing for now - can be enhanced with NLP later
        params = {
            "game": "MortalKombatII-Genesis",  # Default game
            "save_state": "tournament",         # Default state
            "scenario": None,
            "players": 1,
            "timesteps": 100000,
            "learning_rate": 0.001,
            "batch_size": 64
        }
        
        # Parse game name if specified
        if "game:" in user_input:
            game_part = user_input.split("game:")[1].split()[0]
            params["game"] = game_part
            
        # Parse state if specified
        if "state:" in user_input:
            state_part = user_input.split("state:")[1].split()[0]
            params["save_state"] = state_part
            
        # Parse players if specified
        if "players:" in user_input:
            try:
                players = int(user_input.split("players:")[1].split()[0])
                params["players"] = min(max(1, players), 2)  # Ensure between 1-2
            except ValueError:
                logging.warning("Invalid players value, using default")
                
        # Parse timesteps if specified
        if "timesteps:" in user_input:
            try:
                timesteps = int(user_input.split("timesteps:")[1].split()[0])
                params["timesteps"] = max(1000, timesteps)  # Minimum 1000 timesteps
            except ValueError:
                logging.warning("Invalid timesteps value, using default")
                
        return params
        
    except Exception as e:
        logging.error(f"Error parsing user input: {e}")
        return {"error": str(e)}
