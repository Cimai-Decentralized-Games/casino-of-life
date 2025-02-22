# casino_of_life_retro/game_wrappers/mk2/mk2_obs.py

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .mk2_const import GameConsts
from .mk2_gamestate import MK2GameState
from .mk2_ai import MK2AISystem
from .mk2_rf import register_functions

NUM_PARAMS = 16  # Adjust this based on your observation space

class MK2ObservationEnv(gym.Wrapper):
    def __init__(self, env, args, num_players, rf_name):
        super().__init__(env)

        low = np.array([-1] * NUM_PARAMS, dtype=np.float32)
        high = np.array([1] * NUM_PARAMS, dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.num_players = num_players
        if num_players == 2:
            self.action_space = gym.spaces.MultiBinary(len(GameConsts.ACTIONS))

        self.game_state = MK2GameState()
        self.ai_sys = MK2AISystem(args, env, None)

        self.rf_name = rf_name
        self.init_function, self.reward_function, self.done_function = register_functions(self.rf_name)

        self.ram_inited = False

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        self.state = np.zeros(NUM_PARAMS, dtype=np.float32)
        self.game_state = MK2GameState()
        self.ram_inited = False

        return self.state, info

    def step(self, action):
        if self.num_players == 2:
            p2_action = np.zeros_like(action)
            action = np.concatenate([action, p2_action])

        obs, reward, terminated, truncated, info = self.env.step(action)

        if not self.ram_inited:
            self.init_function(self.env)
            self.ram_inited = True

        self.game_state.begin_frame(info)
        
        # Calculate Reward and check if episode is done
        reward = self.reward_function(self.game_state)
        terminated = self.done_function(self.game_state)
       
        self.game_state.end_frame()

        # Update the state based on game_state
        self.state = np.array([
            self.game_state.normalized_p1_x,
            self.game_state.normalized_p1_y,
            self.game_state.normalized_p2_x,
            self.game_state.normalized_p2_y,
            self.game_state.normalized_p1_health,
            self.game_state.normalized_p2_health,
            self.game_state.normalized_p1_stamina,
            self.game_state.normalized_p2_stamina,
            self.game_state.normalized_distance,
            self.game_state.normalized_p1_state,
            self.game_state.normalized_p2_state,
            self.game_state.normalized_p1_action,
            self.game_state.normalized_p2_action,
            self.game_state.normalized_round_timer,
            self.game_state.normalized_p1_rounds_won,
            self.game_state.normalized_p2_rounds_won
        ], dtype=np.float32)

        obs = self.state
        
        return obs, reward, terminated, truncated, info

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]