# casino_of_life_retro/client_bridge/action_mappers.py
import numpy as np
import logging
from typing import List, Optional

class ActionMapper:
    """ Class for mapping actions to retro game actions. """
    def __init__(self, game_controls, game_name: str):
        """
            Args:
                game_controls: A dictionary of all controls.
                game_name: The game that you are trying to map.
        """
        if game_name not in game_controls:
            raise ValueError(f"Game '{game_name}' is not supported.")
        self.game_controls = game_controls[game_name]
        self.default_action = self.game_controls.get("default_action", None) #Default to None

    def map_agent_action(self, agent_actions: List[str]) -> np.ndarray:
        """Maps high-level agent action to Retro's action space."""
        if not self.game_controls:
            raise ValueError("Game controls are not initialized properly")

        if "actions" not in self.game_controls:
            raise KeyError("Action map is not in the game controls.")

        combined_action = np.zeros(len(self.game_controls["actions"][list(self.game_controls["actions"].keys())[0]]), dtype=np.uint8)

        for agent_action in agent_actions:
          if agent_action not in self.game_controls["actions"]:
              if self.default_action:
                combined_action = np.array(self.default_action, dtype=np.uint8)
              else:
                logging.warning(f"Action: '{agent_action}' is not in the game controls, returning a 0.0 action.")
                combined_action = np.array([0] * len(self.game_controls["actions"][list(self.game_controls["actions"].keys())[0]]), dtype=np.uint8)
                return combined_action
          else:
            combined_action = combined_action | np.array(self.game_controls["actions"][agent_action], dtype=np.uint8)

        return combined_action


    def map_raw_action(self, raw_action: np.ndarray) -> str:
        """Maps Retro's action space back to high-level agent action."""
        if not self.game_controls:
           raise ValueError("Game controls are not initialized properly")
        if "actions" not in self.game_controls:
            raise KeyError("Action map is not in the game controls.")

        for action_name, action_array in self.game_controls["actions"].items():
            if np.array_equal(raw_action, action_array):
                return action_name
        return "neutral"  # If no match, return neutral action