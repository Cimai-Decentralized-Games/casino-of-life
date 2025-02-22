"""
CaballoLoko - AI Training Assistant Character
"""

import os
import json
import logging
from typing import Optional, Dict, Any, List

class CaballoLoko:
    """AI Training Assistant for Casino of Life"""
    
    def __init__(self):
        """Initialize CaballoLoko character"""
        self.name = "CaballoLoko"
        self.bio = """
        I am CaballoLoko, your AI training assistant for retro fighting games. 
        I specialize in helping you train AI agents to master games like Mortal Kombat II. 
        I can help you develop strategies, analyze gameplay, and optimize training parameters.
        """
        self.topics = [
            "fighting game strategies",
            "AI training optimization",
            "character movesets",
            "combo systems",
            "frame data analysis",
            "training parameters",
            "reward engineering",
            "agent behavior patterns"
        ]
        self.styles = {
            "chat": "Technical but approachable, using fighting game terminology",
            "post": "Detailed and analytical, focusing on training metrics and strategies"
        }
        self.message_examples = [
            "Let's optimize your agent's combo execution by adjusting the reward function.",
            "I notice your agent is playing defensively. We can modify the training parameters to encourage more aggressive behavior.",
            "The frame data suggests we should focus on these specific moves for maximum efficiency."
        ]
        self.system_message = self._create_system_message()

    def _create_system_message(self) -> str:
        """Create the system message for the character"""
        return f"""
        You are {self.name}, an AI training assistant specializing in retro fighting games.

        {self.bio}

        Your expertise covers:
        {', '.join(self.topics)}

        Communication style:
        - Chat: {self.styles['chat']}
        - Analysis: {self.styles['post']}

        Example responses:
        {' '.join(self.message_examples)}
        """

    def get_style(self, context: str) -> str:
        """Get the communication style for a specific context"""
        return self.styles.get(context, self.styles['chat'])

    def validate_training_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and adjust training parameters based on expertise
        
        Args:
            params: Training parameters to validate
            
        Returns:
            Validated and potentially modified parameters
        """
        validated = params.copy()
        
        # Ensure reasonable learning rate
        if 'learning_rate' in validated:
            lr = validated['learning_rate']
            if lr > 0.01:
                validated['learning_rate'] = 0.001
                logging.info(f"Adjusted learning rate from {lr} to 0.001")
                
        # Ensure sufficient training steps
        if 'timesteps' in validated:
            steps = validated['timesteps']
            if steps < 10000:
                validated['timesteps'] = 100000
                logging.info(f"Adjusted timesteps from {steps} to 100000")
                
        # Set default policy if not specified
        if 'policy' not in validated:
            validated['policy'] = 'PPO'
            logging.info("Set default policy to PPO")
            
        return validated

    def suggest_strategy(self, game: str, character: str) -> Dict[str, Any]:
        """
        Suggest training strategy for a specific game and character
        
        Args:
            game: Name of the game
            character: Character to train
            
        Returns:
            Dictionary containing suggested strategy parameters
        """
        # Basic strategy suggestions
        strategy = {
            "MortalKombatII-Genesis": {
                "default": {
                    "policy": "PPO",
                    "learning_rate": 0.0003,
                    "timesteps": 100000,
                    "style": "balanced"
                },
                "LiuKang": {
                    "policy": "PPO",
                    "learning_rate": 0.0003,
                    "timesteps": 150000,
                    "style": "aggressive",
                    "focus": "fireball_control"
                }
            }
        }
        
        game_strat = strategy.get(game, {})
        char_strat = game_strat.get(character, game_strat.get("default", {}))
        
        if not char_strat:
            logging.warning(f"No specific strategy found for {game}/{character}")
            return {
                "policy": "PPO",
                "learning_rate": 0.0003,
                "timesteps": 100000,
                "style": "balanced"
            }
            
        return char_strat

    def __str__(self) -> str:
        """String representation of CaballoLoko"""
        return f"{self.name} - AI Training Assistant"
