# /casino-of-life/casino-of-life/src/client_bridge/retro_api.py
import gymnasium as gym
from casino_of_life.src.game_environments.retro_env_loader import RetroEnv

class RetroAPI:
    def __init__(self, game=None, state=None, scenario=None, players=1):
        self.game = game
        self.state = state
        self.scenario = scenario
        self.players = players
        self.env = None

    def make_env(self, game=None, state=None, scenario=None, players=None) -> gym.Env:
        """ Wraps retro.make with the custom retro env """
        try:
            # Use passed parameters or fall back to instance variables
            game = game or self.game
            state = state or self.state
            scenario = scenario or self.scenario
            players = players or self.players

            self.env = RetroEnv(
                game=game,
                state=state,
                scenario=scenario,
                players=players
            )
            return self.env
        except Exception as e:
            raise ValueError(f"Could not create Retro environment: {e}") from e
