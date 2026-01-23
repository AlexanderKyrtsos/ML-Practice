"""
Pygame-based renderer for the Snake game.
Handles all visual display and human input.
"""

import pygame
from typing import Optional, Tuple
from snake_game import GameState, Direction


# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 200, 0)
DARK_GREEN = (0, 150, 0)
RED = (200, 0, 0)
GRAY = (40, 40, 40)
BLUE = (0, 100, 200)


class SnakeRenderer:
    """Renders the Snake game using Pygame."""

    def __init__(
        self,
        width: int,
        height: int,
        cell_size: int = 30,
        fps: int = 10,
        title: str = "Snake AI",
    ):
        """
        Initialize the renderer.

        Args:
            width: Grid width in cells
            height: Grid height in cells
            cell_size: Size of each cell in pixels
            fps: Target frames per second
            title: Window title
        """
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.fps = fps

        # Calculate window size (extra space for score bar)
        self.score_bar_height = 40
        self.window_width = width * cell_size
        self.window_height = height * cell_size + self.score_bar_height

        # Initialize Pygame
        pygame.init()
        pygame.display.set_caption(title)
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)

    def render(self, state: GameState, info: Optional[dict] = None) -> None:
        """
        Render the current game state.

        Args:
            state: Current game state
            info: Optional dict with extra info to display (e.g., episode, epsilon)
        """
        # Clear screen
        self.screen.fill(BLACK)

        # Draw grid
        self._draw_grid()

        # Draw food
        self._draw_cell(state.food, RED)

        # Draw snake
        for i, segment in enumerate(state.snake):
            if i == 0:
                # Head
                self._draw_cell(segment, GREEN)
                self._draw_eyes(segment, state.direction)
            else:
                # Body
                self._draw_cell(segment, DARK_GREEN)

        # Draw score bar
        self._draw_score_bar(state, info)

        # Update display
        pygame.display.flip()

        # Control frame rate
        self.clock.tick(self.fps)

    def _draw_grid(self) -> None:
        """Draw the grid lines."""
        for x in range(0, self.window_width, self.cell_size):
            pygame.draw.line(
                self.screen,
                GRAY,
                (x, self.score_bar_height),
                (x, self.window_height),
            )
        for y in range(self.score_bar_height, self.window_height, self.cell_size):
            pygame.draw.line(self.screen, GRAY, (0, y), (self.window_width, y))

    def _draw_cell(self, pos: Tuple[int, int], color: Tuple[int, int, int]) -> None:
        """Draw a filled cell at the given grid position."""
        x, y = pos
        rect = pygame.Rect(
            x * self.cell_size + 1,
            y * self.cell_size + self.score_bar_height + 1,
            self.cell_size - 2,
            self.cell_size - 2,
        )
        pygame.draw.rect(self.screen, color, rect, border_radius=5)

    def _draw_eyes(self, pos: Tuple[int, int], direction: Direction) -> None:
        """Draw eyes on the snake head."""
        x, y = pos
        center_x = x * self.cell_size + self.cell_size // 2
        center_y = y * self.cell_size + self.score_bar_height + self.cell_size // 2

        eye_size = 4
        eye_offset = 6

        if direction == Direction.UP:
            eye1 = (center_x - eye_offset, center_y - eye_offset)
            eye2 = (center_x + eye_offset, center_y - eye_offset)
        elif direction == Direction.DOWN:
            eye1 = (center_x - eye_offset, center_y + eye_offset)
            eye2 = (center_x + eye_offset, center_y + eye_offset)
        elif direction == Direction.LEFT:
            eye1 = (center_x - eye_offset, center_y - eye_offset)
            eye2 = (center_x - eye_offset, center_y + eye_offset)
        else:  # RIGHT
            eye1 = (center_x + eye_offset, center_y - eye_offset)
            eye2 = (center_x + eye_offset, center_y + eye_offset)

        pygame.draw.circle(self.screen, WHITE, eye1, eye_size)
        pygame.draw.circle(self.screen, WHITE, eye2, eye_size)

    def _draw_score_bar(self, state: GameState, info: Optional[dict]) -> None:
        """Draw the score bar at the top."""
        # Background
        pygame.draw.rect(
            self.screen, GRAY, (0, 0, self.window_width, self.score_bar_height)
        )

        # Score
        score_text = self.font.render(f"Score: {state.score}", True, WHITE)
        self.screen.blit(score_text, (10, 8))

        # Additional info
        if info:
            info_parts = []
            if "episode" in info:
                info_parts.append(f"Ep: {info['episode']}")
            if "epsilon" in info:
                info_parts.append(f"Eps: {info['epsilon']:.2f}")
            if "high_score" in info:
                info_parts.append(f"High: {info['high_score']}")

            if info_parts:
                info_text = self.small_font.render(" | ".join(info_parts), True, WHITE)
                self.screen.blit(info_text, (self.window_width - info_text.get_width() - 10, 12))

        # Game over message
        if state.game_over:
            game_over_text = self.font.render("GAME OVER - Press R to restart", True, RED)
            text_rect = game_over_text.get_rect(center=(self.window_width // 2, self.score_bar_height // 2))
            self.screen.blit(game_over_text, text_rect)

    def get_human_action(self) -> Tuple[Optional[Direction], bool, bool]:
        """
        Get action from keyboard input.

        Returns:
            Tuple of (direction, quit_requested, restart_requested)
        """
        direction = None
        quit_requested = False
        restart_requested = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_requested = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP or event.key == pygame.K_w:
                    direction = Direction.UP
                elif event.key == pygame.K_DOWN or event.key == pygame.K_s:
                    direction = Direction.DOWN
                elif event.key == pygame.K_LEFT or event.key == pygame.K_a:
                    direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                    direction = Direction.RIGHT
                elif event.key == pygame.K_r:
                    restart_requested = True
                elif event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    quit_requested = True

        return direction, quit_requested, restart_requested

    def close(self) -> None:
        """Close the renderer and clean up Pygame."""
        pygame.quit()

    def set_fps(self, fps: int) -> None:
        """Change the frame rate."""
        self.fps = fps


class HeadlessRenderer:
    """
    A no-op renderer for headless training.
    Has the same interface as SnakeRenderer but does nothing.
    """

    def __init__(self, *args, **kwargs):
        pass

    def render(self, state: GameState, info: Optional[dict] = None) -> None:
        pass

    def get_human_action(self) -> Tuple[Optional[Direction], bool, bool]:
        return None, False, False

    def close(self) -> None:
        pass

    def set_fps(self, fps: int) -> None:
        pass
