"""
PPO training script for the Mario environment.
"""

import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from wrappers import make_mario_env


def make_env():
    def _init():
        return make_mario_env(render_mode=None)
    return _init


def main():
    log_dir = "./mario_logs"
    checkpoint_dir = "./mario_checkpoints"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    env = DummyVecEnv([make_env()])
    env = VecTransposeImage(env)

    model = PPO(
        "CnnPolicy",
        env,
        learning_rate=2.5e-4,
        n_steps=128,
        batch_size=256,
        n_epochs=4,
        gamma=0.99,
        clip_range=0.2,
        verbose=1,
        tensorboard_log=log_dir,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=100_000,
        save_path=checkpoint_dir,
        name_prefix="mario_ppo",
    )

    total_timesteps = 2_000_000
    print(f"Starting training for {total_timesteps} timesteps...")
    print(f"TensorBoard logs: {log_dir}")
    print(f"Checkpoints: {checkpoint_dir}")
    print("Run: tensorboard --logdir mario_logs")

    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_cb,
        progress_bar=True,
    )

    model.save("mario_ppo_final")
    print("Training complete. Model saved to mario_ppo_final.zip")
    env.close()


if __name__ == "__main__":
    main()
