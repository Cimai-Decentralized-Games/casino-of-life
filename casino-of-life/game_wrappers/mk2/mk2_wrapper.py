# casino_of_life_retro/game_wrappers/mk2/mk2_wrapper.py

import gym
import numpy as np

class MK2Wrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env

    def reset(self):
        obs = self.env.reset()
        return self.process_obs(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        processed_obs = self.process_obs(obs)
        processed_reward = self.process_reward(reward, info)
        return processed_obs, processed_reward, done, info

    def process_obs(self, obs):
        # Process the observation (game screen) if needed
        # For example, you might want to crop the screen, normalize values, etc.
        return obs

    def reset(self):
        obs = self.env.reset()
        # Modify initial observation if needed
        return obs