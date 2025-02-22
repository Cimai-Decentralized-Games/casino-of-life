from typing import Dict, Any, Optional
import json
import logging
import os

DEFAULT_CHARACTER_PATH = os.path.join(os.path.dirname(__file__), 'character.json')

class Character:
    def __init__(self, json_path: str = DEFAULT_CHARACTER_PATH):
        """
        Initializes a Character object from a JSON file.
        
        Args:
            json_path (str): Path to the character JSON file.
        """
        self.data = self.load_character_data(json_path)
        
        # Load character identity first
        self.character_id = self.data.get("character", "CaballoLoko")
        self.name = self.data.get("name", self.character_id)
        
        # Rest of the initialization
        self.plugins = self.data.get("plugins", [])
        self.model_provider = self.data.get("modelProvider", None)
        self.clients = self.data.get("clients", [])
        self.bio = self.data.get("bio", "No bio available.")
        self.lore = self.data.get("lore", [])
        self.knowledge = self.data.get("knowledge", [])
        self.message_examples = self.data.get("messageExamples", [])
        self.post_examples = self.data.get("postExamples", [])
        self.people = self.data.get("people", [])
        self.topics = self.data.get("topics", [])
        self.style = self.data.get("style", {})
        self.adjectives = self.data.get("adjectives", [])
        self.settings = self.data.get("settings", {})
        
        # Training-specific attributes
        self.training_dialogue = self.data.get("training_dialogue", {})
        self.training_configs = self.data.get("training_configs", {})
        self.current_training_state = None
        self.training_context = {}

    def load_character_data(self, json_path: str) -> Dict[str, Any]:
        """
        Loads character data from a JSON file.
        
        Args:
            json_path (str): Path to the JSON file.
            
        Returns:
            Dict containing character data.
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                logging.info(f"Loaded character data for: {data.get('character', 'unknown')}")
                return data
        except Exception as e:
            logging.error(f"Error loading character data: {e}")
            return {}

    def get_training_response(self, user_input: str) -> str:
        """
        Generates a training-specific response based on the conversation state.
        
        Args:
            user_input (str): The user's message
            
        Returns:
            str: Character's response
        """
        if not self.current_training_state:
            self.current_training_state = "initial"
            return self.training_dialogue["initial_greeting"]

        user_input_lower = user_input.lower()
        
        if self.current_training_state == "initial":
            # Extract character selection
            for character in ["sub-zero", "scorpion", "liu-kang"]:  # Add all valid characters
                if character in user_input_lower:
                    self.training_context["character"] = character
                    self.current_training_state = "strategy"
                    return self.training_dialogue["strategy_question"]

        elif self.current_training_state == "strategy":
            # Extract strategy selection
            for strategy in self.training_configs["strategies"]:
                if strategy in user_input_lower:
                    self.training_context["strategy"] = strategy
                    recommended_policy = self.training_configs["strategies"][strategy]["recommended_policy"]
                    self.training_context["policy"] = recommended_policy
                    
                    response = self.training_dialogue["policy_explanation"].format(
                        strategy=strategy,
                        policy=recommended_policy,
                        reason=self.training_configs["policies"][recommended_policy]
                    )
                    self.current_training_state = "save_state"
                    return response

        elif self.current_training_state == "save_state":
            # Handle save state selection
            self.training_context["save_state"] = "fight_start"  # Default for now
            params = self.training_configs["strategies"][self.training_context["strategy"]]["default_params"]
            self.training_context.update(params)
            
            return self.training_dialogue["confirmation"].format(**self.training_context)

        # Default response if no specific state matches
        return "I'm not sure I understood. Could you please clarify?"

    def get_training_context(self) -> Dict[str, Any]:
        """
        Returns the current training configuration.
        
        Returns:
            Dict containing the training parameters
        """
        return self.training_context

    def reset_training_state(self):
        """
        Resets the training conversation state.
        """
        self.current_training_state = None
        self.training_context = {}

    def get_style(self, key):
        return self.style.get(key, [])

    def __repr__(self):
        """Returns a string representation of the Character object."""
        return f"<Character: {self.name}>"
