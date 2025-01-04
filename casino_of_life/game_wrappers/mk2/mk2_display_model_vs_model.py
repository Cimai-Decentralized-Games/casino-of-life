# casino_of_life_retro/game_wrappers/mk2/mk2_display_model_vs_model.py

import os
import gymnasium as gym
import numpy as np
import pygame
import pygame.freetype

from .mk2_const import GameConsts
from .mk2_gamestate import MK2GameState

class MK2ModelVsModelDisplayEnv(gym.Wrapper):
    def __init__(self, env, args, model1_desc, model2_desc, model1_params, model2_params, action_names):
        super().__init__(env)

        self.FB_WIDTH = args.display_width
        self.FB_HEIGHT = args.display_height

        self.GAME_WIDTH = GameConsts.SCREEN_WIDTH * 3
        self.GAME_HEIGHT = GameConsts.SCREEN_HEIGHT * 3

        # Set up display positions
        self.setup_display_positions()

        # Init Window
        pygame.init()
        self.screen = pygame.display.set_mode((self.FB_WIDTH, self.FB_HEIGHT))
        self.main_surf = pygame.Surface((self.FB_WIDTH, self.FB_HEIGHT))
        self.main_surf.set_colorkey((0,0,0))

        # Set up fonts
        self.setup_fonts()

        self.args = args
        self.action_names = action_names
        self.model1_desc = model1_desc
        self.model2_desc = model2_desc
        self.model1_params = model1_params
        self.model2_params = model2_params
        self.p1_action_probabilities = [0] * len(action_names)
        self.p2_action_probabilities = [0] * len(action_names)

        self.game_state = MK2GameState()

    def setup_display_positions(self):
        self.BASIC_INFO_X = (self.FB_WIDTH >> 1) - 50
        self.BASIC_INFO_Y = self.GAME_HEIGHT + 10
        self.AP_X = self.GAME_WIDTH + 100
        self.AP_Y = 200
        self.MODELDESC1_X = 20
        self.MODELDESC1_Y = self.FB_HEIGHT - 60
        self.NUM_PARAMS1_X = self.MODELDESC1_X + 200
        self.NUM_PARAMS1_Y = self.FB_HEIGHT - 60
        self.MODELDESC2_X = self.FB_WIDTH - 220
        self.MODELDESC2_Y = self.FB_HEIGHT - 60
        self.NUM_PARAMS2_X = self.MODELDESC2_X - 200
        self.NUM_PARAMS2_Y = self.FB_HEIGHT - 60
        self.VS_X = (self.FB_WIDTH >> 1) - 50
        self.VS_Y = self.FB_HEIGHT - 100

    def setup_fonts(self):
        self.font = pygame.freetype.SysFont('Arial', 30)
        self.info_font = pygame.freetype.SysFont('Arial', 20)
        self.info_font_big = pygame.freetype.SysFont('Arial', 40)
        self.vs_font = pygame.freetype.SysFont('Arial', 80)

    def draw_string(self, font, text, pos, color):
        text_rect = font.get_rect(text)
        text_rect.topleft = pos
        font.render_to(self.main_surf, text_rect.topleft, text, color)
        return text_rect.bottom

    def draw_action_probabilities(self, pos_x, pos_y, action_probabilities):
        y = pos_y
        for action, prob in zip(self.action_names, action_probabilities):
            self.draw_string(self.info_font, f"{action}: {prob:.4f}", (pos_x, y), (255, 255, 255))
            y += 25

    def draw_basic_info(self):
        self.draw_string(self.vs_font, 'VS', (self.VS_X, self.VS_Y), (255, 0, 0))
        self.draw_string(self.font, self.args.env, (self.VS_X - 100, self.FB_HEIGHT - 30), (255, 255, 255))

        # Model 1
        self.draw_string(self.info_font, 'MODEL', (self.MODELDESC1_X, self.MODELDESC1_Y), (0, 255, 0))
        self.draw_string(self.info_font, 'PARAMETERS', (self.NUM_PARAMS1_X, self.NUM_PARAMS1_Y), (0, 255, 0))
        self.draw_string(self.info_font_big, self.model1_desc, (self.MODELDESC1_X, self.MODELDESC1_Y - 40), (255, 255, 255))
        self.draw_string(self.info_font_big, f"{self.model1_params}", (self.NUM_PARAMS1_X, self.NUM_PARAMS1_Y - 40), (255, 255, 255))

        # Model 2
        self.draw_string(self.info_font, 'MODEL', (self.MODELDESC2_X, self.MODELDESC2_Y), (0, 255, 0))
        self.draw_string(self.info_font, 'PARAMETERS', (self.NUM_PARAMS2_X, self.NUM_PARAMS2_Y), (0, 255, 0))
        self.draw_string(self.info_font_big, self.model2_desc, (self.MODELDESC2_X, self.MODELDESC2_Y - 40), (255, 255, 255))
        self.draw_string(self.info_font_big, f"{self.model2_params}", (self.NUM_PARAMS2_X, self.NUM_PARAMS2_Y - 40), (255, 255, 255))

    def draw_game_state(self):
        # Draw health bars
        p1_health_width = int(self.game_state.p1_health * 2)
        p2_health_width = int(self.game_state.p2_health * 2)
        pygame.draw.rect(self.main_surf, (255, 0, 0), (20, 20, 200, 20))
        pygame.draw.rect(self.main_surf, (0, 255, 0), (20, 20, p1_health_width, 20))
        pygame.draw.rect(self.main_surf, (255, 0, 0), (self.FB_WIDTH - 220, 20, 200, 20))
        pygame.draw.rect(self.main_surf, (0, 255, 0), (self.FB_WIDTH - 220, 20, p2_health_width, 20))

        # Draw round wins
        self.draw_string(self.info_font_big, f"Rounds: {self.game_state.p1_rounds_won}", (20, 50), (255, 255, 255))
        self.draw_string(self.info_font_big, f"Rounds: {self.game_state.p2_rounds_won}", (self.FB_WIDTH - 220, 50), (255, 255, 255))

        # Draw timer
        self.draw_string(self.info_font_big, f"Time: {self.game_state.round_timer}", (self.FB_WIDTH // 2 - 50, 20), (255, 255, 0))

    def draw_frame(self, frame_img):
        self.main_surf.fill((0, 0, 0))
        
        # Draw game screen
        game_screen = pygame.surfarray.make_surface(np.transpose(frame_img, (1, 0, 2)))
        game_x = (self.FB_WIDTH - self.GAME_WIDTH) // 2
        self.main_surf.blit(pygame.transform.scale(game_screen, (self.GAME_WIDTH, self.GAME_HEIGHT)), (game_x, 0))

        self.draw_basic_info()
        self.draw_game_state()
        self.draw_action_probabilities(20, 100, self.p1_action_probabilities)
        self.draw_action_probabilities(self.FB_WIDTH - 220, 100, self.p2_action_probabilities)

        self.screen.blit(self.main_surf, (0, 0))
        pygame.display.flip()

    def process_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    return False
        return True

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.game_state.reset()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        self.game_state.update(info)
        
        frame = self.env.render(mode='rgb_array')
        self.draw_frame(frame)

        if not self.process_input():
            return obs, reward, True, truncated, info

        return obs, reward, terminated, truncated, info

    def set_action_probabilities(self, p1_probs, p2_probs):
        self.p1_action_probabilities = p1_probs
        self.p2_action_probabilities = p2_probs

    def close(self):
        pygame.quit()
        return self.env.close()