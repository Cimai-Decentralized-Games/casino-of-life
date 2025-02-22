# casino_of_life_retro/game_wrappers/mk2/mk2_display.py

import pygame
import pygame.freetype
import numpy as np
import gymnasium as gym
from .mk2_const import GameConsts
from .mk2_gamestate import MK2GameState

class MK2GameDisplayEnv(gym.Wrapper):
    def __init__(self, env, args, model1_desc, model2_desc, model1_params, model2_params, action_names):
        super().__init__(env)
        
        self.setup_display(args)
        self.setup_fonts()
        
        self.args = args
        self.action_names = action_names
        self.model1_desc, self.model2_desc = model1_desc, model2_desc
        self.model1_params, self.model2_params = model1_params, model2_params
        
        self.game_state = MK2GameState()
        self.reward_history = [0.0] * 200
        
        self.p1_action_probs = [0] * len(action_names)
        self.p2_action_probs = [0] * len(action_names)

    def setup_display(self, args):
        self.FB_WIDTH, self.FB_HEIGHT = args.display_width, args.display_height
        self.GAME_WIDTH, self.GAME_HEIGHT = GameConsts.SCREEN_WIDTH * 3, GameConsts.SCREEN_HEIGHT * 3
        
        pygame.init()
        self.screen = pygame.display.set_mode((self.FB_WIDTH, self.FB_HEIGHT))
        self.main_surf = pygame.Surface((self.FB_WIDTH, self.FB_HEIGHT))

    def setup_fonts(self):
        self.font = pygame.freetype.SysFont('Arial', 30)
        self.info_font = pygame.freetype.SysFont('Arial', 20)
        self.info_font_big = pygame.freetype.SysFont('Arial', 50)

    def draw_string(self, font, text, pos, color):
        font.render_to(self.main_surf, pos, text, color)

    def draw_game_frame(self, frame):
        game_surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        scaled_surf = pygame.transform.scale(game_surf, (self.GAME_WIDTH, self.GAME_HEIGHT))
        self.main_surf.blit(scaled_surf, ((self.FB_WIDTH - self.GAME_WIDTH) // 2, 0))

    def draw_health_bars(self):
        p1_health = int(self.game_state.p1_health * 2)
        p2_health = int(self.game_state.p2_health * 2)
        pygame.draw.rect(self.main_surf, (255, 0, 0), (20, 20, 200, 20))
        pygame.draw.rect(self.main_surf, (0, 255, 0), (20, 20, p1_health, 20))
        pygame.draw.rect(self.main_surf, (255, 0, 0), (self.FB_WIDTH - 220, 20, 200, 20))
        pygame.draw.rect(self.main_surf, (0, 255, 0), (self.FB_WIDTH - 220, 20, p2_health, 20))

    def draw_game_info(self):
        self.draw_string(self.info_font, f"Round: {self.game_state.current_round}", (self.FB_WIDTH // 2 - 50, 20), (255, 255, 255))
        self.draw_string(self.info_font, f"Time: {self.game_state.round_timer}", (self.FB_WIDTH // 2 - 50, 50), (255, 255, 255))
        self.draw_string(self.info_font, f"P1 Wins: {self.game_state.p1_rounds_won}", (20, 50), (255, 255, 255))
        self.draw_string(self.info_font, f"P2 Wins: {self.game_state.p2_rounds_won}", (self.FB_WIDTH - 120, 50), (255, 255, 255))

    def draw_action_probabilities(self):
        y = self.GAME_HEIGHT + 20
        for action, p1_prob, p2_prob in zip(self.action_names, self.p1_action_probs, self.p2_action_probs):
            self.draw_string(self.info_font, f"{action}: {p1_prob:.2f}", (20, y), (255, 255, 255))
            self.draw_string(self.info_font, f"{action}: {p2_prob:.2f}", (self.FB_WIDTH - 220, y), (255, 255, 255))
            y += 25

    def draw_reward_histogram(self):
        pygame.draw.line(self.main_surf, (255,255,255), (20, self.FB_HEIGHT - 100), (620, self.FB_HEIGHT - 100))
        for i, r in enumerate(self.reward_history):
            height = abs(r) * 50
            y = self.FB_HEIGHT - 100 if r < 0 else self.FB_HEIGHT - 100 - height
            pygame.draw.rect(self.main_surf, (0,255,0), (20 + i * 3, y, 2, height))

    def draw_frame(self, frame):
        self.main_surf.fill((30, 30, 30))
        self.draw_game_frame(frame)
        self.draw_health_bars()
        self.draw_game_info()
        self.draw_action_probabilities()
        self.draw_reward_histogram()
        self.screen.blit(self.main_surf, (0, 0))
        pygame.display.flip()

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.game_state.update(info)
        self.reward_history.append(reward)
        self.reward_history = self.reward_history[1:]
        frame = self.env.render(mode='rgb_array')
        self.draw_frame(frame)
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.game_state.reset()
        return obs, info

    def set_action_probabilities(self, p1_probs, p2_probs):
        self.p1_action_probs = p1_probs
        self.p2_action_probs = p2_probs

    def close(self):
        pygame.quit()
        return self.env.close()