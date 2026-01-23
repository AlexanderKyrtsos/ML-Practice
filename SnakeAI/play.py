"""
Play Snake - either as a human or watch the AI play.
"""

import argparse
from typing import Optional

from snake_game import SnakeGame, Direction
from renderer import SnakeRenderer
from agent import DQNAgent
from environment import SnakeEnv


def play_human(grid_size: int = 20, fps: int = 10) -> None:
    """
    Play Snake with keyboard controls.

    Controls:
        Arrow keys or WASD: Move
        R: Restart
        Q or Escape: Quit
    """
    game = SnakeGame(width=grid_size, height=grid_size)
    renderer = SnakeRenderer(grid_size, grid_size, fps=fps, title="Snake - Human Player")

    high_score = 0
    running = True

    print("\nSnake Game - Human Player")
    print("=" * 30)
    print("Controls:")
    print("  Arrow keys or WASD: Move")
    print("  R: Restart")
    print("  Q or Escape: Quit")
    print("=" * 30)

    while running:
        state = game.get_state()
        high_score = max(high_score, state.score)

        # Render
        renderer.render(state, {"high_score": high_score})

        # Get input
        direction, quit_requested, restart_requested = renderer.get_human_action()

        if quit_requested:
            running = False
        elif restart_requested:
            game.reset()
        elif not state.game_over:
            game.step(direction)

    renderer.close()
    print(f"\nFinal high score: {high_score}")


def play_ai(
    model_path: str,
    grid_size: int = 20,
    fps: int = 10,
    num_games: int = 0,
    device: Optional[str] = None,
) -> None:
    """
    Watch the AI play Snake.

    Args:
        model_path: Path to trained model
        grid_size: Grid size
        fps: Frames per second
        num_games: Number of games (0 = infinite)
        device: Device to use
    """
    # Load agent
    agent = DQNAgent.from_file(model_path, device=device)
    agent.epsilon = 0  # No exploration

    # Create environment and renderer
    env = SnakeEnv(width=grid_size, height=grid_size)
    renderer = SnakeRenderer(grid_size, grid_size, fps=fps, title="Snake - AI Player")

    games_played = 0
    total_score = 0
    high_score = 0
    running = True

    print(f"\nSnake Game - AI Player")
    print(f"Model: {model_path}")
    print("=" * 30)
    print("Controls:")
    print("  R: Restart")
    print("  Q or Escape: Quit")
    print("  +/-: Adjust speed")
    print("=" * 30)

    state = env.reset()

    while running:
        game_state = env.get_state()
        high_score = max(high_score, game_state.score)

        # Render
        renderer.render(game_state, {
            "episode": games_played + 1,
            "high_score": high_score,
        })

        # Check for user input
        _, quit_requested, restart_requested = renderer.get_human_action()

        if quit_requested:
            running = False
        elif restart_requested or game_state.game_over:
            if game_state.game_over:
                games_played += 1
                total_score += game_state.score
                print(f"Game {games_played}: Score {game_state.score}")

                if num_games > 0 and games_played >= num_games:
                    running = False
                    continue

            state = env.reset()
        else:
            # AI selects action
            action = agent.select_action(state, training=False)
            state, _, _, _ = env.step(action)

    renderer.close()

    if games_played > 0:
        avg_score = total_score / games_played
        print(f"\n{'=' * 30}")
        print(f"Games played: {games_played}")
        print(f"Average score: {avg_score:.2f}")
        print(f"High score: {high_score}")


def benchmark_ai(
    model_path: str,
    grid_size: int = 20,
    num_games: int = 100,
    device: Optional[str] = None,
) -> None:
    """
    Benchmark the AI without rendering (fast).

    Args:
        model_path: Path to trained model
        grid_size: Grid size
        num_games: Number of games to play
        device: Device to use
    """
    import time

    # Load agent
    agent = DQNAgent.from_file(model_path, device=device)
    agent.epsilon = 0

    # Create environment
    env = SnakeEnv(width=grid_size, height=grid_size)

    scores = []
    steps_list = []
    start_time = time.time()

    print(f"\nBenchmarking AI on {num_games} games...")
    print(f"Model: {model_path}")
    print(f"Grid size: {grid_size}x{grid_size}")
    print("-" * 40)

    for game in range(num_games):
        state = env.reset()
        done = False

        while not done:
            action = agent.select_action(state, training=False)
            state, _, done, info = env.step(action)

        scores.append(info["score"])
        steps_list.append(info["steps"])

        if (game + 1) % 10 == 0:
            print(f"Completed {game + 1}/{num_games} games...")

    elapsed = time.time() - start_time

    print("\n" + "=" * 40)
    print("BENCHMARK RESULTS")
    print("=" * 40)
    print(f"Games played:     {num_games}")
    print(f"Time elapsed:     {elapsed:.2f}s")
    print(f"Games per second: {num_games / elapsed:.1f}")
    print("-" * 40)
    print(f"Average score:    {sum(scores) / len(scores):.2f}")
    print(f"Max score:        {max(scores)}")
    print(f"Min score:        {min(scores)}")
    print(f"Median score:     {sorted(scores)[len(scores)//2]}")
    print("-" * 40)
    print(f"Average steps:    {sum(steps_list) / len(steps_list):.1f}")
    print(f"Max steps:        {max(steps_list)}")


def main():
    parser = argparse.ArgumentParser(description="Play Snake")
    subparsers = parser.add_subparsers(dest="mode", help="Play mode")

    # Human mode
    human_parser = subparsers.add_parser("human", help="Play as human")
    human_parser.add_argument("--grid-size", type=int, default=20, help="Grid size")
    human_parser.add_argument("--fps", type=int, default=10, help="Game speed (FPS)")

    # AI mode
    ai_parser = subparsers.add_parser("ai", help="Watch AI play")
    ai_parser.add_argument("model", type=str, help="Path to trained model")
    ai_parser.add_argument("--grid-size", type=int, default=20, help="Grid size")
    ai_parser.add_argument("--fps", type=int, default=10, help="Game speed (FPS)")
    ai_parser.add_argument("--games", type=int, default=0, help="Number of games (0=infinite)")
    ai_parser.add_argument("--device", type=str, default=None, help="Device (cpu/cuda)")

    # Benchmark mode
    bench_parser = subparsers.add_parser("benchmark", help="Benchmark AI (headless)")
    bench_parser.add_argument("model", type=str, help="Path to trained model")
    bench_parser.add_argument("--grid-size", type=int, default=20, help="Grid size")
    bench_parser.add_argument("--games", type=int, default=100, help="Number of games")
    bench_parser.add_argument("--device", type=str, default=None, help="Device (cpu/cuda)")

    args = parser.parse_args()

    if args.mode == "human":
        play_human(grid_size=args.grid_size, fps=args.fps)
    elif args.mode == "ai":
        play_ai(
            model_path=args.model,
            grid_size=args.grid_size,
            fps=args.fps,
            num_games=args.games,
            device=args.device,
        )
    elif args.mode == "benchmark":
        benchmark_ai(
            model_path=args.model,
            grid_size=args.grid_size,
            num_games=args.games,
            device=args.device,
        )
    else:
        # Default to human mode
        play_human()


if __name__ == "__main__":
    main()
