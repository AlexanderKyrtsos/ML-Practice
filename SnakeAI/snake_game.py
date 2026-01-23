"""
Core Snake game logic - no rendering dependencies.
This module contains the pure game mechanics.
"""

import random
from enum import IntEnum
from typing import List, Tuple, Optional
from dataclasses import dataclass, field


class Direction(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


# Movement vectors for each direction
DIRECTION_VECTORS = {
    Direction.UP: (0, -1),
    Direction.RIGHT: (1, 0),
    Direction.DOWN: (0, 1),
    Direction.LEFT: (-1, 0),
}

# Opposite directions (can't reverse into yourself)
OPPOSITE_DIRECTIONS = {
    Direction.UP: Direction.DOWN,
    Direction.DOWN: Direction.UP,
    Direction.LEFT: Direction.RIGHT,
    Direction.RIGHT: Direction.LEFT,
}


@dataclass
class GameState:
    """Represents the complete state of a Snake game."""
    snake: List[Tuple[int, int]]  # Head is first element
    direction: Direction
    food: Tuple[int, int]
    score: int
    game_over: bool
    width: int
    height: int
    steps_since_food: int = 0
    total_steps: int = 0

    @property
    def head(self) -> Tuple[int, int]:
        return self.snake[0]

    @property
    def tail(self) -> List[Tuple[int, int]]:
        return self.snake[1:]


class SnakeGame:
    """
    Core Snake game implementation.

    The game grid uses (x, y) coordinates where:
    - x increases to the right (0 to width-1)
    - y increases downward (0 to height-1)
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
        Initialize a Snake game.

        Args:
            width: Grid width in cells
            height: Grid height in cells
            initial_length: Starting snake length
            max_steps_without_food: Max steps before game over (prevents infinite loops)
                                   If None, defaults to width * height * 2
            seed: Random seed for reproducibility
        """
        self.width = width
        self.height = height
        self.initial_length = initial_length
        self.max_steps_without_food = max_steps_without_food or (width * height * 2)

        if seed is not None:
            random.seed(seed)

        self.reset()

    def reset(self) -> GameState:
        """Reset the game to initial state and return it."""
        # Start snake in the middle, going right
        center_x = self.width // 2
        center_y = self.height // 2

        self.snake = [
            (center_x - i, center_y) for i in range(self.initial_length)
        ]
        self.direction = Direction.RIGHT
        self.score = 0
        self.game_over = False
        self.steps_since_food = 0
        self.total_steps = 0

        # Place initial food
        self.food = self._place_food()

        return self.get_state()

    def _place_food(self) -> Tuple[int, int]:
        """Place food in a random empty cell."""
        empty_cells = [
            (x, y)
            for x in range(self.width)
            for y in range(self.height)
            if (x, y) not in self.snake
        ]

        if not empty_cells:
            # Snake fills the entire grid - player wins!
            return self.snake[-1]  # Return tail position (will be eaten)

        return random.choice(empty_cells)

    def step(self, action: Optional[Direction] = None) -> Tuple[GameState, float, bool]:
        """
        Advance the game by one step.

        Args:
            action: New direction to move, or None to continue current direction.
                   Invalid moves (reversing) are ignored.

        Returns:
            Tuple of (new_state, reward, done)
        """
        if self.game_over:
            return self.get_state(), 0.0, True

        # Update direction (prevent reversing)
        if action is not None and action != OPPOSITE_DIRECTIONS[self.direction]:
            self.direction = action

        # Calculate new head position
        dx, dy = DIRECTION_VECTORS[self.direction]
        old_head = self.snake[0]
        head_x, head_y = old_head
        new_head = (head_x + dx, head_y + dy)

        # Check for collisions
        reward = 0.0

        # Wall collision
        if not (0 <= new_head[0] < self.width and 0 <= new_head[1] < self.height):
            self.game_over = True
            reward = -10.0
            return self.get_state(), reward, True

        # Self collision (check against body, not including tail which will move)
        if new_head in self.snake[:-1]:
            self.game_over = True
            reward = -10.0
            return self.get_state(), reward, True

        # Move snake
        self.snake.insert(0, new_head)
        self.total_steps += 1
        self.steps_since_food += 1

        # Check for food
        if new_head == self.food:
            self.score += 1
            self.steps_since_food = 0
            reward = 10.0
            self.food = self._place_food()
            # Don't remove tail - snake grows
        else:
            self.snake.pop()
            # Reward shaping: encourage moving toward food
            old_dist = abs(old_head[0] - self.food[0]) + abs(old_head[1] - self.food[1])
            new_dist = abs(new_head[0] - self.food[0]) + abs(new_head[1] - self.food[1])
            if new_dist < old_dist:
                reward = 0.1  # Small reward for getting closer
            else:
                reward = -0.1  # Small penalty for moving away

        # Check for timeout (prevents infinite loops during training)
        if self.steps_since_food >= self.max_steps_without_food:
            self.game_over = True
            reward = -5.0

        return self.get_state(), reward, self.game_over

    def get_state(self) -> GameState:
        """Return the current game state."""
        return GameState(
            snake=self.snake.copy(),
            direction=self.direction,
            food=self.food,
            score=self.score,
            game_over=self.game_over,
            width=self.width,
            height=self.height,
            steps_since_food=self.steps_since_food,
            total_steps=self.total_steps,
        )

    def get_observation(self) -> List[float]:
        """
        Get a feature vector representing the current state.
        Used as input for the neural network.

        Returns 24 features:
        - 3 immediate danger indicators (straight, right, left)
        - 3 distance to danger (normalized, in each relative direction)
        - 4 direction indicators (current direction one-hot)
        - 4 food direction indicators (food left/right/up/down relative to head)
        - 4 tail direction indicators (where is tail relative to head)
        - 3 available space in each direction (flood fill, normalized)
        - 3 body segment proximity (how close is body in each direction)
        """
        head_x, head_y = self.snake[0]
        max_dist = max(self.width, self.height)

        # Relative directions
        dir_straight = self.direction
        dir_right = Direction((self.direction + 1) % 4)
        dir_left = Direction((self.direction - 1) % 4)

        # Get positions in each direction
        point_straight = self._get_point_in_direction(dir_straight)
        point_right = self._get_point_in_direction(dir_right)
        point_left = self._get_point_in_direction(dir_left)

        # Immediate danger detection
        danger_straight = self._is_collision(point_straight)
        danger_right = self._is_collision(point_right)
        danger_left = self._is_collision(point_left)

        # Distance to danger in each direction (how many steps until collision)
        dist_straight = self._distance_to_collision(dir_straight) / max_dist
        dist_right = self._distance_to_collision(dir_right) / max_dist
        dist_left = self._distance_to_collision(dir_left) / max_dist

        # Current direction (one-hot)
        dir_up = self.direction == Direction.UP
        dir_right_abs = self.direction == Direction.RIGHT
        dir_down = self.direction == Direction.DOWN
        dir_left_abs = self.direction == Direction.LEFT

        # Food direction (relative to head)
        food_left = self.food[0] < head_x
        food_right = self.food[0] > head_x
        food_up = self.food[1] < head_y
        food_down = self.food[1] > head_y

        # Tail direction (helps avoid circling back into self)
        tail_x, tail_y = self.snake[-1]
        tail_left = tail_x < head_x
        tail_right = tail_x > head_x
        tail_up = tail_y < head_y
        tail_down = tail_y > head_y

        # Available space in each direction (simplified flood fill)
        space_straight = self._count_reachable_space(point_straight) / (self.width * self.height)
        space_right = self._count_reachable_space(point_right) / (self.width * self.height)
        space_left = self._count_reachable_space(point_left) / (self.width * self.height)

        # Closest body segment in each direction
        body_dist_straight = self._nearest_body_distance(dir_straight) / max_dist
        body_dist_right = self._nearest_body_distance(dir_right) / max_dist
        body_dist_left = self._nearest_body_distance(dir_left) / max_dist

        return [
            # Immediate danger (3)
            float(danger_straight),
            float(danger_right),
            float(danger_left),
            # Distance to danger (3)
            dist_straight,
            dist_right,
            dist_left,
            # Current direction (4)
            float(dir_up),
            float(dir_right_abs),
            float(dir_down),
            float(dir_left_abs),
            # Food location (4)
            float(food_left),
            float(food_right),
            float(food_up),
            float(food_down),
            # Tail location (4)
            float(tail_left),
            float(tail_right),
            float(tail_up),
            float(tail_down),
            # Available space (3)
            space_straight,
            space_right,
            space_left,
            # Body proximity (3)
            body_dist_straight,
            body_dist_right,
            body_dist_left,
        ]

    def _distance_to_collision(self, direction: Direction) -> int:
        """Count steps until collision in a direction."""
        dx, dy = DIRECTION_VECTORS[direction]
        x, y = self.snake[0]
        distance = 0

        while True:
            x += dx
            y += dy
            if not (0 <= x < self.width and 0 <= y < self.height):
                break
            if (x, y) in self.snake:
                break
            distance += 1

        return distance

    def _count_reachable_space(self, start: Tuple[int, int], max_count: int = 50) -> int:
        """
        Count reachable empty cells from a starting point (limited flood fill).
        Returns 0 if start is a collision.
        """
        if self._is_collision(start):
            return 0

        visited = set()
        queue = [start]
        count = 0

        while queue and count < max_count:
            x, y = queue.pop(0)
            if (x, y) in visited:
                continue
            visited.add((x, y))
            count += 1

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.width and 0 <= ny < self.height and
                    (nx, ny) not in visited and (nx, ny) not in self.snake):
                    queue.append((nx, ny))

        return count

    def _nearest_body_distance(self, direction: Direction) -> int:
        """Find distance to nearest body segment in a direction."""
        dx, dy = DIRECTION_VECTORS[direction]
        x, y = self.snake[0]
        distance = 0
        max_dist = max(self.width, self.height)

        while distance < max_dist:
            x += dx
            y += dy
            distance += 1
            if not (0 <= x < self.width and 0 <= y < self.height):
                return max_dist  # Hit wall first
            if (x, y) in self.snake:
                return distance

        return max_dist

    def _get_point_in_direction(self, direction: Direction) -> Tuple[int, int]:
        """Get the point one step in the given direction from the head."""
        dx, dy = DIRECTION_VECTORS[direction]
        head_x, head_y = self.snake[0]
        return (head_x + dx, head_y + dy)

    def _is_collision(self, point: Tuple[int, int]) -> bool:
        """Check if a point would cause a collision."""
        x, y = point
        # Wall collision
        if not (0 <= x < self.width and 0 <= y < self.height):
            return True
        # Self collision
        if point in self.snake:
            return True
        return False

    def get_valid_actions(self) -> List[Direction]:
        """Get list of valid actions (non-suicidal moves)."""
        valid = []
        for direction in Direction:
            if direction == OPPOSITE_DIRECTIONS[self.direction]:
                continue
            point = self._get_point_in_direction(direction)
            if not self._is_collision(point):
                valid.append(direction)
        return valid if valid else [self.direction]  # If all moves are bad, go straight


if __name__ == "__main__":
    # Simple test
    game = SnakeGame(width=10, height=10, seed=42)
    state = game.get_state()
    print(f"Initial state: snake={state.snake}, food={state.food}")

    # Play a few random moves
    for i in range(10):
        action = random.choice(list(Direction))
        state, reward, done = game.step(action)
        print(f"Step {i+1}: action={action.name}, reward={reward:.1f}, score={state.score}, done={done}")
        if done:
            break
