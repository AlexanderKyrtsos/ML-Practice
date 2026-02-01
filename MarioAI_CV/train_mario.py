"""
PPO training script for the Mario environment.
"""

import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecTransposeImage, VecFrameStack, VecNormalize
from wrappers import make_mario_env


class MaxXPosCallback(BaseCallback):
    """Logs the max x_pos reached across all envs each rollout."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._max_x = 0.0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            x = info.get("x_pos", 0.0)
            if x > self._max_x:
                self._max_x = x
        return True

    def _on_rollout_end(self) -> None:
        self.logger.record("mario/max_x_pos", self._max_x)
        self._max_x = 0.0


NUM_ENVS = 32


def make_env(rank):
    def _init():
        return make_mario_env(render_mode=None, training_mode=True)
    return _init


def linear_schedule(initial_value: float):
    """Linear decay from initial_value to 0."""
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


def main():
    log_dir = "./mario_logs"
    checkpoint_dir = "./mario_checkpoints"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    env = SubprocVecEnv([make_env(i) for i in range(NUM_ENVS)])
    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0)
    env = VecTransposeImage(env)
    env = VecFrameStack(env, n_stack=4)

    model = PPO(
        "CnnPolicy",
        env,
        learning_rate=linear_schedule(1e-4),
        n_steps=128,
        batch_size=256,
        n_epochs=3,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.05,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=log_dir,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=250_000 // NUM_ENVS,
        save_path=checkpoint_dir,
        name_prefix="mario_ppo",
    )

    total_timesteps = 10_000_000
    print(f"Starting training for {total_timesteps} timesteps with {NUM_ENVS} parallel envs...")
    print(f"TensorBoard logs: {log_dir}")
    print(f"Checkpoints: {checkpoint_dir}")
    print("Run: tensorboard --logdir mario_logs")

    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_cb, MaxXPosCallback()],
        progress_bar=False,
    )

    model.save("mario_ppo_final")
    print("Training complete. Model saved to mario_ppo_final.zip")
    env.close()


if __name__ == "__main__":
    main()
