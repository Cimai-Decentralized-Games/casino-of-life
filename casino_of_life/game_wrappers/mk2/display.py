# casino_of_life_retro/game_wrappers/display.py

import pygame
import pygame.freetype
import numpy as np
import cv2
import gymnasium as gym

FB_WIDTH = 1920
FB_HEIGHT = 1080

class MK2DisplayEnv(gym.Wrapper):
    def __init__(self, env, args, model1_desc, model2_desc, model1_params, model2_params, button_names):
        super().__init__(env)
        
        self.GAME_WIDTH = 320 * 4
        self.GAME_HEIGHT = 240 * 4
        
        pygame.init()
        self.screen = pygame.display.set_mode((args.display_width, args.display_height))
        self.main_surf = pygame.Surface((FB_WIDTH, FB_HEIGHT))
        self.main_surf.set_colorkey((0,0,0))
        
        self.font = pygame.freetype.SysFont('Arial', 30)
        self.info_font = pygame.freetype.SysFont('Arial', 20)
        self.info_font_big = pygame.freetype.SysFont('Arial', 50)
        self.vs_font = pygame.freetype.SysFont('Arial', 80)
        
        self.args = args
        self.button_names = button_names
        self.model1_desc = model1_desc
        self.model2_desc = model2_desc
        self.model1_params = model1_params
        self.model2_params = model2_params
        
        self.p1_action_probabilities = [0] * len(button_names)
        self.p2_action_probabilities = [0] * len(button_names)
        
        self.setup_positions()

    def setup_positions(self):
        self.VS_X = (FB_WIDTH >> 1) - 50
        self.VS_Y = FB_HEIGHT - 100
        self.MODELDESC1_X = (FB_WIDTH - self.GAME_WIDTH) >> 1
        self.MODELDESC1_Y = FB_HEIGHT - 20
        self.NUM_PARAMS1_X = self.MODELDESC1_X + 200
        self.NUM_PARAMS1_Y = FB_HEIGHT - 20
        self.MODELDESC2_X = FB_WIDTH - ((FB_WIDTH - self.GAME_WIDTH) >> 1) - 50
        self.MODELDESC2_Y = FB_HEIGHT - 20
        self.NUM_PARAMS2_X = self.MODELDESC2_X - 350
        self.NUM_PARAMS2_Y = FB_HEIGHT - 20

    def draw_string(self, font, text, pos, color):
        text_rect = font.get_rect(text)
        text_rect.topleft = pos
        font.render_to(self.main_surf, text_rect.topleft, text, color)
        return text_rect.bottom

    def draw_basic_info(self):
        self.draw_string(self.vs_font, 'VS', (self.VS_X, self.VS_Y), (0, 255, 0))
        self.draw_string(self.font, self.args.env, (self.VS_X - 100, FB_HEIGHT - 30), (255, 255, 255))

        # Model 1
        self.draw_string(self.info_font, 'MODEL', (self.MODELDESC1_X, self.MODELDESC1_Y), (0, 255, 0))
        self.draw_string(self.info_font, 'NUM PARAMETERS', (self.NUM_PARAMS1_X, self.NUM_PARAMS1_Y), (0, 255, 0))
        self.draw_string(self.info_font_big, self.model1_desc, (self.MODELDESC1_X, self.MODELDESC1_Y - 60), (255, 255, 255))
        self.draw_string(self.info_font_big, f'{self.model1_params}', (self.NUM_PARAMS1_X, self.NUM_PARAMS1_Y - 60), (255, 255, 255))

        # Model 2
        self.draw_string(self.info_font, 'MODEL', (self.MODELDESC2_X, self.MODELDESC2_Y), (0, 255, 0))
        self.draw_string(self.info_font, 'NUM PARAMETERS', (self.NUM_PARAMS2_X, self.NUM_PARAMS2_Y), (0, 255, 0))
        self.draw_string(self.info_font_big, self.model2_desc, (self.MODELDESC2_X, self.MODELDESC2_Y - 60), (255, 255, 255))
        self.draw_string(self.info_font_big, f'{self.model2_params}', (self.NUM_PARAMS2_X, self.NUM_PARAMS2_Y - 60), (255, 255, 255))

    def draw_action_probabilities(self, pos_x, pos_y, action_probabilities):
        y = pos_y + 10
        for button, prob in zip(self.button_names, action_probabilities):
            self.draw_string(self.font, f"{button}: {prob:.4f}", (pos_x, y), (255, 255, 255))
            y += 30

    def draw_frame(self, frame_img):
        self.main_surf.fill((0, 0, 0))
        emu_screen = np.transpose(frame_img, (1,0,2))
        surf = pygame.surfarray.make_surface(emu_screen)
        game_x = (FB_WIDTH - self.GAME_WIDTH) >> 1
        self.main_surf.blit(pygame.transform.scale(surf, (self.GAME_WIDTH, self.GAME_HEIGHT)), (game_x, 0))

        self.draw_basic_info()
        self.draw_action_probabilities(0, 100, self.p1_action_probabilities)
        self.draw_action_probabilities(self.GAME_WIDTH + game_x, 100, self.p2_action_probabilities)

        self.screen.blit(pygame.transform.smoothscale(self.main_surf, (self.args.display_width, self.args.display_height)), (0, 0))
        pygame.display.flip()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.render()
        return obs, reward, done, info

    def render(self, mode='human'):
        if mode == 'human':
            frame = self.env.render(mode='rgb_array')
            self.draw_frame(frame)
        return self.env.render(mode)

    def close(self):
        pygame.quit()
        return self.env.close()

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def set_action_probabilities(self, p1_probs, p2_probs):
        self.p1_action_probabilities = p1_probs
        self.p2_action_probabilities = p2_probs
