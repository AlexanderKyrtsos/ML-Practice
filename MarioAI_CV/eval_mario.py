"""
Evaluation script — load a trained model and watch it play.
"""

import sys
import pygame
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from wrappers import make_mario_env


def make_env():
    def _init():
        return make_mario_env(render_mode="human")
    return _init


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "mario_ppo_final"
    print(f"Loading model from {model_path}...")

    env = DummyVecEnv([make_env()])
    env = VecTransposeImage(env)

    model = PPO.load(model_path, env=env)

    obs = env.reset()
    total_reward = 0
    episodes = 0

    while episodes < 5:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward[0]

        # Render is handled by the env (render_mode="human")
        env.envs[0].render()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                env.close()
                return

        if done[0]:
            episodes += 1
            print(f"Episode {episodes}: reward={total_reward:.1f}, info={info[0]}")
            total_reward = 0
            obs = env.reset()

    env.close()


if __name__ == "__main__":
    main()
