"""
Observation preprocessing wrappers for the Mario environment.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2


class GrayScaleObservation(gym.ObservationWrapper):
    """Convert RGB observation to grayscale."""

    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(*obs_shape, 1), dtype=np.uint8
        )

    def observation(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        return np.expand_dims(gray, axis=-1)


class ResizeObservation(gym.ObservationWrapper):
    """Resize observation to (size, size)."""

    def __init__(self, env, size=84):
        super().__init__(env)
        self.size = size
        channels = self.observation_space.shape[-1]
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(size, size, channels), dtype=np.uint8
        )

    def observation(self, obs):
        resized = cv2.resize(obs, (self.size, self.size), interpolation=cv2.INTER_AREA)
        if resized.ndim == 2:
            resized = np.expand_dims(resized, axis=-1)
        return resized


class NormalizeObservation(gym.ObservationWrapper):
    """Normalize pixel values to [0, 1]."""

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=self.observation_space.shape,
            dtype=np.float32,
        )

    def observation(self, obs):
        return obs.astype(np.float32) / 255.0


class FrameSkip(gym.Wrapper):
    """Repeat action for `skip` frames, return last observation."""

    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


class FrameStack(gym.Wrapper):
    """Stack `num_stack` consecutive frames along the last axis."""

    def __init__(self, env, num_stack=4):
        super().__init__(env)
        self._num_stack = num_stack
        self._frames = []
        low = np.repeat(self.observation_space.low, num_stack, axis=-1)
        high = np.repeat(self.observation_space.high, num_stack, axis=-1)
        self.observation_space = spaces.Box(
            low=low, high=high, dtype=self.observation_space.dtype
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._frames = [obs] * self._num_stack
        return np.concatenate(self._frames, axis=-1), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._frames.append(obs)
        self._frames = self._frames[-self._num_stack:]
        stacked = np.concatenate(self._frames, axis=-1)
        return stacked, reward, terminated, truncated, info


def make_mario_env(render_mode=None, training_mode=False, max_episode_steps=3000):
    """Create a fully wrapped Mario environment for RL training.

    FrameStack is not applied here; use VecFrameStack in the training script.
    NormalizeObservation is not applied; SB3 CnnPolicy handles normalization.
    """
    from mario_env import MarioEnv
    env = MarioEnv(render_mode=render_mode, training_mode=training_mode)
    env = FrameSkip(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, size=84)
    if max_episode_steps is not None:
        from gymnasium.wrappers import TimeLimit
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    return env
