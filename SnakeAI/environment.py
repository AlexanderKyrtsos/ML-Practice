"""
Gym-like environment wrapper for Snake.
Provides a clean interface for reinforcement learning training.
"""

from typing import Tuple, List, Optional
import numpy as np

from snake_game import SnakeGame, Direction, GameState


class SnakeEnv:
    """
    Gym-like environment for Snake.

    Actions:
        0: Continue straight
        1: Turn right (relative to current direction)
        2: Turn left (relative to current direction)

    Observation:
        11-dimensional vector (see SnakeGame.get_observation)
    """

    def __init__(
        self,
        width: int = 20,
        height: int = 20,
        initial_length: int = 3,
        max_steps_without_food: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize the environment.

        Args:
            width: Grid width
            height: Grid height
            initial_length: Starting snake length
            max_steps_without_food: Steps before timeout
            seed: Random seed
        """
        self.game = SnakeGame(
            width=width,
            height=height,
            initial_length=initial_length,
            max_steps_without_food=max_steps_without_food,
            seed=seed,
        )

        # Action and observation spaces info
        self.n_actions = 3  # straight, right, left
        self.observation_size = 24

    def reset(self) -> np.ndarray:
        """Reset the environment and return initial observation."""
        self.game.reset()
        return np.array(self.game.get_observation(), dtype=np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Take a step in the environment.

        Args:
            action: 0=straight, 1=turn right, 2=turn left

        Returns:
            Tuple of (observation, reward, done, info)
        """
        # Convert relative action to absolute direction
        current_dir = self.game.direction
        if action == 0:
            # Go straight
            new_direction = current_dir
        elif action == 1:
            # Turn right
            new_direction = Direction((current_dir + 1) % 4)
        else:
            # Turn left
            new_direction = Direction((current_dir - 1) % 4)

        # Take step
        state, reward, done = self.game.step(new_direction)

        # Get observation
        obs = np.array(self.game.get_observation(), dtype=np.float32)

        # Build info dict
        info = {
            "score": state.score,
            "snake_length": len(state.snake),
            "steps": state.total_steps,
        }

        return obs, reward, done, info

    def get_state(self) -> GameState:
        """Get the current game state (for rendering)."""
        return self.game.get_state()

    def render(self) -> None:
        """Placeholder for compatibility - rendering is handled separately."""
        pass

    def close(self) -> None:
        """Clean up resources."""
        pass


class VectorizedSnakeEnv:
    """
    Vectorized environment that runs multiple Snake games in parallel.
    Useful for faster training with batched updates.
    """

    def __init__(
        self,
        n_envs: int = 4,
        width: int = 20,
        height: int = 20,
        initial_length: int = 3,
        max_steps_without_food: Optional[int] = None,
    ):
        """
        Initialize vectorized environment.

        Args:
            n_envs: Number of parallel environments
            width: Grid width
            height: Grid height
            initial_length: Starting snake length
            max_steps_without_food: Steps before timeout
        """
        self.n_envs = n_envs
        self.envs = [
            SnakeEnv(
                width=width,
                height=height,
                initial_length=initial_length,
                max_steps_without_food=max_steps_without_food,
            )
            for _ in range(n_envs)
        ]

        self.n_actions = 3
        self.observation_size = 11

    def reset(self) -> np.ndarray:
        """Reset all environments."""
        observations = [env.reset() for env in self.envs]
        return np.stack(observations)

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict]]:
        """
        Take a step in all environments.

        Args:
            actions: Array of actions, one per environment

        Returns:
            Tuple of (observations, rewards, dones, infos)
        """
        observations = []
        rewards = []
        dones = []
        infos = []

        for i, (env, action) in enumerate(zip(self.envs, actions)):
            obs, reward, done, info = env.step(int(action))

            # Auto-reset on done
            if done:
                obs = env.reset()

            observations.append(obs)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)

        return (
            np.stack(observations),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=bool),
            infos,
        )

    def close(self) -> None:
        """Clean up all environments."""
        for env in self.envs:
            env.close()
