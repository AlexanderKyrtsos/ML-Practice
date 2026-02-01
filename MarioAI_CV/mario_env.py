"""
Gymnasium-compatible environment wrapper for the Mario game.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from mario_game import MarioGame


class MarioEnv(gym.Env):
    """
    Gymnasium environment wrapping the Mario game.

    Action space: Discrete(7)
        0=NOOP, 1=right, 2=right+jump, 3=right+run, 4=right+run+jump, 5=jump, 6=left

    Observation space: Box(0, 255, (240, 256, 3), uint8) — raw RGB pixels
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None, training_mode=False):
        super().__init__()
        self.render_mode = render_mode
        self.game = MarioGame(
            render_mode=(render_mode == "human"),
            training_mode=training_mode,
        )

        self.action_space = spaces.Discrete(7)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(240, 256, 3), dtype=np.uint8
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = self.game.reset()
        return obs, self.game._get_info()

    def step(self, action):
        obs, reward, done, info = self.game.step(int(action))
        if info.get("timed_out", False):
            terminated = False
            truncated = True
        else:
            terminated = done
            truncated = False
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            self.game.render()
        elif self.render_mode == "rgb_array":
            return self.game._get_obs()

    def close(self):
        pygame.quit()
