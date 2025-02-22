import gymnasium as gym
import retro
import numpy as np
import gc
from pathlib import Path
from typing import Optional
from casino_of_life.utils.config import DATA_DIR  # Updated import path

# Match the C++ constants
N_BUTTONS = 16
MAX_PLAYERS = 2

class StochasticFrameSkip(gym.Wrapper):
    def __init__(self, env, n, stickprob):
        gym.Wrapper.__init__(self, env)
        self.n = n
        self.stickprob = stickprob
        self.curac = None
        self.rng = np.random.RandomState()

    def reset(self, **kwargs):
        self.curac = None
        return self.env.reset(**kwargs)

    def step(self, ac):
        done = False
        totrew = 0
        for i in range(self.n):
            if self.curac is None:
                self.curac = ac
            elif i == 0:
                if self.rng.rand() > self.stickprob:
                    self.curac = ac
            elif i == 1:
                self.curac = ac
            ob, rew, terminated, truncated, info = self.env.step(self.curac)
            totrew += rew
            if terminated or truncated:
                done = True
                break
        return ob, totrew, terminated, truncated, info

def make_retro(*, game, state=retro.State.DEFAULT, num_players=1):
    """Create a retro environment using hardcoded button count"""
    if num_players > MAX_PLAYERS:
        raise ValueError(f"Maximum {MAX_PLAYERS} players supported")
        
    # Force cleanup
    gc.collect()
    
    # Create environment with hardcoded button count
    env = retro.make(
        game=game,
        state=state,
        players=num_players,
        obs_type=retro.Observations.IMAGE,
        use_restricted_actions=retro.Actions.FILTERED
    )
    return env

class RetroEnv(gym.Env):
    def __init__(self, game: str, state: Optional[str] = retro.State.DEFAULT, 
                 scenario: Optional[str] = None, players: int = 1):
        super().__init__()
        
        if players > MAX_PLAYERS:
            raise ValueError(f"Maximum {MAX_PLAYERS} players supported")
            
        # Create environment
        self.env = make_retro(game=game, state=state, num_players=players)
        
        # Apply wrappers in same order as mk2_envs
        if True:  # use_frame_skip
            self.env = StochasticFrameSkip(self.env, n=4, stickprob=0.25)
        self.env = gym.wrappers.ResizeObservation(self.env, (84, 84))
        self.env = gym.wrappers.GrayScaleObservation(self.env)
        self.env = gym.wrappers.FrameStack(self.env, 4)
        
        # Get spaces from wrapped env
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def step(self, action):
        return self.env.step(action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def render(self):
        return self.env.render()

    def close(self):
        if hasattr(self, 'env'):
            self.env.close()
        gc.collect()
