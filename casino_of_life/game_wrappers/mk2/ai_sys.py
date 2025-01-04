
"""
AI System for Mortal Kombat II
"""

from .models import load_model_based_on_extension, predict_action
from .models import get_num_parameters, get_model_probabilities

class AISys:
    def __init__(self, args, env, logger):
        self.args = args
        self.env = env
        self.logger = logger
        self.model = None
        self.use_model = True
        
        # for display
        self.display_probs = None
        self.model_num_params = None

    def set_model(self, model_path):
        if model_path:
            self.model = load_model_based_on_extension(model_path)
            self.model_num_params = get_num_parameters(self.model)
            self.logger.info(f"Model loaded from {model_path}")
            self.logger.info(f"Number of model parameters: {self.model_num_params}")

    def predict(self, state, info, deterministic):
        if self.model:
            action = predict_action(self.model, state)
            self.display_probs = get_model_probabilities(self.model, state)
            return action
        else:
            self.logger.warning("No model set. Returning None for action.")
            return None

    def get_model_info(self):
        return {
            "num_parameters": self.model_num_params,
            "display_probs": self.display_probs
        }