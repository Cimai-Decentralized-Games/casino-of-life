"""
Action Mappers Module - Maps high-level actions to game-specific controls
"""

from typing import List, Dict, Any, Optional
import logging

class ActionMapper:
    """Maps high-level actions to game-specific button combinations"""
    
    def __init__(self, game_controls: Dict[str, Any], game: str):
        """
        Initialize action mapper for a specific game
        
        Args:
            game_controls: Dictionary mapping actions to button combinations
            game: Name of the game
        """
        self.game_controls = game_controls
        self.game = game
        self.button_map = self._get_button_map()

    def _get_button_map(self) -> Dict[str, List[str]]:
        """Get the button mapping for the current game"""
        if self.game not in self.game_controls:
            logging.warning(f"No specific controls found for {self.game}, using defaults")
            return self._get_default_controls()
        return self.game_controls[self.game]

    def _get_default_controls(self) -> Dict[str, List[str]]:
        """Get default control mappings"""
        return {
            "punch": ["B"],
            "kick": ["A"],
            "block": ["Y"],
            "jump": ["UP"],
            "crouch": ["DOWN"],
            "forward": ["RIGHT"],
            "backward": ["LEFT"],
            "special1": ["DOWN", "RIGHT", "B"],
            "special2": ["DOWN", "LEFT", "A"],
            "fatality": ["DOWN", "DOWN", "UP", "B"]
        }

    def map_agent_action(self, actions: List[str]) -> List[int]:
        """
        Map high-level actions to game-specific button presses
        
        Args:
            actions: List of high-level actions to perform
            
        Returns:
            List of button combinations as integers
        """
        try:
            button_presses = []
            for action in actions:
                if action in self.button_map:
                    button_presses.extend(self.button_map[action])
                else:
                    logging.warning(f"Unknown action {action} for game {self.game}")
            
            # Convert button names to integers (0 or 1)
            # This assumes the game's button order matches the emulator's expectations
            button_states = [0] * 12  # Standard number of buttons
            for button in button_presses:
                button_idx = self._get_button_index(button)
                if button_idx is not None:
                    button_states[button_idx] = 1
                    
            return button_states
            
        except Exception as e:
            logging.error(f"Error mapping action: {e}")
            return [0] * 12  # Return no buttons pressed on error

    def _get_button_index(self, button: str) -> Optional[int]:
        """Get the index for a button name"""
        button_indices = {
            "B": 0,
            "A": 1,
            "Y": 2,
            "X": 3,
            "L": 4,
            "R": 5,
            "UP": 6,
            "DOWN": 7,
            "LEFT": 8,
            "RIGHT": 9,
            "START": 10,
            "SELECT": 11
        }
        return button_indices.get(button.upper())
