"""
Training script for the Snake AI.
Supports both headless training and visualization.
"""

import argparse
import time
from pathlib import Path
from typing import Optional
from collections import deque

import numpy as np

from environment import SnakeEnv
from agent import DQNAgent


def train(
    episodes: int = 1000,
    grid_size: int = 20,
    hidden_size: int = 256,
    learning_rate: float = 0.001,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.01,
    epsilon_decay: float = 0.995,
    batch_size: int = 64,
    buffer_size: int = 100000,
    target_update_freq: int = 100,
    save_path: str = "models/snake_dqn.pth",
    save_freq: int = 100,
    visualize: bool = False,
    visualize_freq: int = 50,
    fps: int = 15,
    device: Optional[str] = None,
    resume: Optional[str] = None,
) -> None:
    """
    Train the Snake AI.

    Args:
        episodes: Number of episodes to train
        grid_size: Size of the grid
        hidden_size: Hidden layer size
        learning_rate: Learning rate
        gamma: Discount factor
        epsilon_start: Initial exploration rate
        epsilon_end: Minimum exploration rate
        epsilon_decay: Epsilon decay per episode
        batch_size: Training batch size
        buffer_size: Replay buffer size
        target_update_freq: Steps between target updates
        save_path: Path to save the model
        save_freq: Episodes between saves
        visualize: Whether to show some episodes
        visualize_freq: Show every N episodes when visualize=True
        fps: Visualization FPS
        device: Device to use
        resume: Path to resume training from
    """
    # Create save directory
    save_dir = Path(save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)

    # Create environment
    env = SnakeEnv(width=grid_size, height=grid_size)

    # Create or load agent
    if resume:
        agent = DQNAgent.from_file(resume, device=device)
        print(f"Resumed from {resume}")
        print(f"Starting at episode {agent.episodes_done}, epsilon {agent.epsilon:.4f}")
    else:
        agent = DQNAgent(
            state_size=env.observation_size,
            action_size=env.n_actions,
            hidden_size=hidden_size,
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay=epsilon_decay,
            buffer_size=buffer_size,
            batch_size=batch_size,
            target_update_freq=target_update_freq,
            device=device,
        )

    # Renderer (only import pygame if needed)
    renderer = None
    if visualize:
        from renderer import SnakeRenderer
        renderer = SnakeRenderer(grid_size, grid_size, fps=fps)

    # Training stats
    scores = deque(maxlen=100)
    best_score = 0
    total_steps = 0
    start_time = time.time()

    print(f"\nStarting training for {episodes} episodes...")
    print(f"Grid size: {grid_size}x{grid_size}")
    print(f"Device: {agent.device}")
    print("-" * 50)

    try:
        for episode in range(1, episodes + 1):
            state = env.reset()
            episode_reward = 0
            episode_steps = 0
            done = False

            # Decide if we're visualizing this episode
            show_episode = visualize and (episode % visualize_freq == 0)

            while not done:
                # Select action
                action = agent.select_action(state, training=True)

                # Take step
                next_state, reward, done, info = env.step(action)

                # Store transition
                agent.store_transition(state, action, reward, next_state, done)

                # Train
                loss = agent.train_step()

                # Update state
                state = next_state
                episode_reward += reward
                episode_steps += 1
                total_steps += 1

                # Visualize if needed
                if show_episode and renderer:
                    game_state = env.get_state()
                    renderer.render(game_state, {
                        "episode": episode,
                        "epsilon": agent.epsilon,
                        "high_score": best_score,
                    })

                    # Handle quit
                    _, quit_requested, _ = renderer.get_human_action()
                    if quit_requested:
                        raise KeyboardInterrupt

            # Episode complete
            score = info["score"]
            scores.append(score)
            avg_score = np.mean(scores)
            best_score = max(best_score, score)

            # Decay epsilon
            agent.decay_epsilon()

            # Log progress
            if episode % 10 == 0:
                elapsed = time.time() - start_time
                eps_per_sec = episode / elapsed if elapsed > 0 else 0
                print(
                    f"Episode {episode:5d} | "
                    f"Score: {score:3d} | "
                    f"Avg(100): {avg_score:6.2f} | "
                    f"Best: {best_score:3d} | "
                    f"Epsilon: {agent.epsilon:.4f} | "
                    f"Steps: {total_steps:7d} | "
                    f"Eps/s: {eps_per_sec:.1f}"
                )

            # Save checkpoint
            if episode % save_freq == 0:
                agent.save(save_path)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")

    finally:
        # Final save
        agent.save(save_path)
        print(f"\nTraining complete!")
        print(f"Final model saved to {save_path}")
        print(f"Best score: {best_score}")
        print(f"Final avg score (last 100): {np.mean(scores):.2f}")

        if renderer:
            renderer.close()


def main():
    parser = argparse.ArgumentParser(description="Train Snake AI")

    # Training parameters
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes")
    parser.add_argument("--grid-size", type=int, default=20, help="Grid size")
    parser.add_argument("--hidden-size", type=int, default=256, help="Hidden layer size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--epsilon-start", type=float, default=1.0, help="Initial epsilon")
    parser.add_argument("--epsilon-end", type=float, default=0.01, help="Final epsilon")
    parser.add_argument("--epsilon-decay", type=float, default=0.995, help="Epsilon decay")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--buffer-size", type=int, default=100000, help="Replay buffer size")
    parser.add_argument("--target-update", type=int, default=100, help="Target network update frequency")

    # Save/load
    parser.add_argument("--save-path", type=str, default="models/snake_dqn.pth", help="Model save path")
    parser.add_argument("--save-freq", type=int, default=100, help="Save frequency (episodes)")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")

    # Visualization
    parser.add_argument("--visualize", action="store_true", help="Show some episodes during training")
    parser.add_argument("--viz-freq", type=int, default=50, help="Visualize every N episodes")
    parser.add_argument("--fps", type=int, default=15, help="Visualization FPS")

    # Hardware
    parser.add_argument("--device", type=str, default=None, help="Device (cpu/cuda/mps)")

    args = parser.parse_args()

    train(
        episodes=args.episodes,
        grid_size=args.grid_size,
        hidden_size=args.hidden_size,
        learning_rate=args.lr,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        target_update_freq=args.target_update,
        save_path=args.save_path,
        save_freq=args.save_freq,
        visualize=args.visualize,
        visualize_freq=args.viz_freq,
        fps=args.fps,
        device=args.device,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
