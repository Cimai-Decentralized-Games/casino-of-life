# casino_of_life_retro/game_wrappers/mk2/mk2_ai.py

import math
import random
from .models import init_model, get_num_parameters, get_model_probabilities
from mk2_const import GameConsts
from mk2_gamestate import MK2GameState

MODEL_NONE = 0
MODEL_DEFENSIVE = 1
MODEL_OFFENSIVE = 2

class MK2AISystem:
    def __init__(self, args, env, logger):
        self.args = args
        self.logger = logger
        self.env = env

        self.game_state = MK2GameState()

        self.models = [None, None, None]
        self.model_params = [None, None, None]
        self.display_probs = [0] * len(GameConsts.ACTIONS)
        self.model_in_use = 0 
        self.num_models = 0

    def set_models(self, model_paths):
        for i, path in enumerate(model_paths):
            if path:
                self.models[i] = init_model(None, path, self.args.alg, self.args, self.env, self.logger)
                self.model_params[i] = get_num_parameters(self.models[i])
                self.num_models += 1
                self.model_in_use = i

    def predict(self, model_index, model_input, deterministic):
        actions = self.models[model_index].predict(model_input, deterministic=deterministic)[0]
        self.display_probs = get_model_probabilities(self.models[model_index], model_input)[0]
        self.model_in_use = model_index
        return actions

    def think_two_models(self, model_input, state, deterministic):
        actions = [0] * len(GameConsts.ACTIONS)

        if state.player_is_attacking:
            actions = self.predict(MODEL_OFFENSIVE, model_input, deterministic)
        elif state.player_is_defending:
            actions = self.predict(MODEL_DEFENSIVE, model_input, deterministic)
        else:
            # Neutral state, use a mix of both models or a default strategy
            actions = self.default_strategy(state)

        return actions

    def default_strategy(self, state):
        actions = [0] * len(GameConsts.ACTIONS)
        # Implement a basic strategy here, e.g., move towards opponent and block
        if state.distance_to_opponent > GameConsts.CLOSE_DISTANCE:
            actions[GameConsts.MOVE_FORWARD] = 1
        else:
            actions[GameConsts.BLOCK] = 1
        return actions

    def predict(self, state, info, deterministic):
        if info is None:
            return [[0] * len(GameConsts.ACTIONS)]
        
        self.game_state.begin_frame(info)
        
        if self.num_models == 1:
            actions = self.predict(self.model_in_use, state, deterministic)
        elif self.models[1] and self.models[2]:
            actions = [self.think_two_models(state, self.game_state, deterministic)]
        else:          
            actions = [self.default_strategy(self.game_state)]

        self.game_state.end_frame()

        return actions