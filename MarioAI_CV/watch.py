"""Watch a trained Mario agent play."""

import sys
import pygame
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack
from wrappers import make_mario_env


def make_env():
    def _init():
        return make_mario_env(render_mode="human", training_mode=False, max_episode_steps=None)
    return _init


def main():
    if len(sys.argv) < 2:
        print("Usage: python watch.py <checkpoint_path> [fps]")
        print("Example: python watch.py mario_checkpoints/mario_ppo_2000000_steps.zip 30")
        sys.exit(1)

    model_path = sys.argv[1]
    fps = int(sys.argv[2]) if len(sys.argv) > 2 else 60

    env = DummyVecEnv([make_env()])
    env = VecTransposeImage(env)
    env = VecFrameStack(env, n_stack=4)

    model = PPO.load(model_path, env=env)
    print(f"Loaded model from {model_path}")

    # Walk the wrapper chain to find the MarioEnv
    inner = env.envs[0]
    while hasattr(inner, 'env'):
        inner = inner.env
    clock = pygame.time.Clock()

    obs = env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        inner.game.render()
        clock.tick(fps)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                env.close()
                return

        if done[0]:
            print(f"Episode done — x_pos: {info[0].get('x_pos', 0):.0f}, "
                  f"score: {info[0].get('score', 0)}, won: {info[0].get('won', False)}")
            obs = env.reset()

    env.close()


if __name__ == "__main__":
    main()
