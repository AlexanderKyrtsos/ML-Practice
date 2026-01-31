"""
Super Mario Bros World 1-1 Clone
Simplified recreation using solid-color rectangles, no image assets.
Exposes a Gym-compatible step() API for RL training.
"""

import pygame
import numpy as np
import sys

from level_data import (
    TILE, EMPTY, GROUND, BRICK, QUESTION, QUESTION_USED,
    PIPE_TL, PIPE_TR, PIPE_BL, PIPE_BR, BLOCK, FLAGPOLE, FLAG_TOP,
    load_level, ensure_default_levels, LEVEL_1_1_PATH,
)

# --- Constants ---
LOGICAL_W, LOGICAL_H = 256, 240
SCALE = 3
WINDOW_W, WINDOW_H = LOGICAL_W * SCALE, LOGICAL_H * SCALE
FPS = 60

# Colors
SKY_BLUE = (107, 140, 255)
GROUND_BROWN = (200, 76, 12)
BRICK_BROWN = (160, 52, 8)
QUESTION_YELLOW = (255, 163, 71)
QUESTION_USED_BROWN = (128, 80, 32)
PIPE_GREEN = (0, 168, 0)
PIPE_GREEN_LIGHT = (40, 200, 40)
BLOCK_GRAY = (130, 130, 130)
FLAGPOLE_WHITE = (200, 200, 200)
FLAG_GREEN = (0, 200, 0)
MARIO_RED = (228, 0, 0)
MARIO_SKIN = (255, 163, 71)
GOOMBA_BROWN = (170, 80, 30)
KOOPA_GREEN = (0, 168, 68)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

TILE_COLORS = {
    GROUND: GROUND_BROWN,
    BRICK: BRICK_BROWN,
    QUESTION: QUESTION_YELLOW,
    QUESTION_USED: QUESTION_USED_BROWN,
    PIPE_TL: PIPE_GREEN_LIGHT,
    PIPE_TR: PIPE_GREEN_LIGHT,
    PIPE_BL: PIPE_GREEN,
    PIPE_BR: PIPE_GREEN,
    BLOCK: BLOCK_GRAY,
    FLAGPOLE: FLAGPOLE_WHITE,
    FLAG_TOP: FLAG_GREEN,
}

# Physics
GRAVITY = 0.6
MAX_FALL = 8.0
MARIO_ACCEL = 0.15
MARIO_DECEL = 0.1
MARIO_MAX_WALK = 1.9
MARIO_MAX_RUN = 3.0
JUMP_VEL = -10.0
JUMP_VEL_SHORT = -5.0  # released early



class Enemy:
    def __init__(self, x, y, kind="goomba"):
        self.x = float(x)
        self.y = float(y)
        self.vx = -0.8
        self.vy = 0.0
        self.kind = kind  # "goomba" or "koopa"
        self.w = 14
        self.h = 14 if kind == "goomba" else 14
        self.alive = True
        self.squish_timer = 0

    def rect(self):
        return pygame.Rect(int(self.x), int(self.y), self.w, self.h)


class MarioGame:
    def __init__(self, render_mode=False, level_path=None):
        self.render_mode = render_mode
        self.level_path = level_path
        self.screen = None
        self.logical_surface = pygame.Surface((LOGICAL_W, LOGICAL_H))
        self.clock = pygame.time.Clock()
        self._load_level_data()
        self.level_w = len(self.level[0])
        self._mario_start = self._loaded_mario_start

        if render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
            pygame.display.set_caption("Super Mario Bros 1-1")
            self.font = pygame.font.SysFont("arial", 8)
        else:
            if not pygame.get_init():
                pygame.init()
                pygame.display.set_mode((1, 1), pygame.NOFRAME)
            self.font = pygame.font.SysFont("arial", 8)

        self.reset()

    def _load_level_data(self):
        ensure_default_levels()
        path = self.level_path or LEVEL_1_1_PATH
        tiles, enemies, mario_start = load_level(path)
        self.level = tiles
        self._loaded_enemies = enemies
        self._loaded_mario_start = mario_start

    def reset(self):
        self._load_level_data()
        self.mario_x = float(self._mario_start[0])
        self.mario_y = float(self._mario_start[1])
        self.mario_vx = 0.0
        self.mario_vy = 0.0
        self.mario_w = 14
        self.mario_h = 16
        self.on_ground = False
        self.jumping = False
        self.jump_held = False
        self.facing_right = True
        self.dead = False
        self.death_timer = 0
        self.won = False

        self.camera_x = 0
        self.score = 0
        self.coins = 0
        self.time = 400
        self.frame_count = 0

        self._spawn_enemies()
        self._prev_x = self.mario_x

        obs = self._get_obs()
        return obs

    def _spawn_enemies(self):
        self.enemies = []
        for x, y, kind in self._loaded_enemies:
            self.enemies.append(Enemy(x, y, kind))

    def _tile_at(self, col, row):
        if 0 <= row < 15 and 0 <= col < self.level_w:
            return self.level[row][col]
        return EMPTY

    def _is_solid(self, tile):
        return tile in (GROUND, BRICK, QUESTION, QUESTION_USED, PIPE_TL, PIPE_TR,
                        PIPE_BL, PIPE_BR, BLOCK, FLAGPOLE)

    def _collide_tile_rect(self, rect):
        """Return list of (row, col, tile) for solid tiles overlapping rect."""
        hits = []
        left_col = max(0, rect.left // TILE)
        right_col = min(self.level_w - 1, rect.right // TILE)
        top_row = max(0, rect.top // TILE)
        bot_row = min(14, rect.bottom // TILE)
        for r in range(top_row, bot_row + 1):
            for c in range(left_col, right_col + 1):
                t = self._tile_at(c, r)
                if self._is_solid(t):
                    hits.append((r, c, t))
        return hits

    def step(self, action):
        """
        action: int 0-6
        0=NOOP, 1=right, 2=right+jump, 3=right+run, 4=right+run+jump, 5=jump, 6=left
        Returns: (observation, reward, done, info)
        """
        if self.dead or self.won:
            return self._get_obs(), 0.0, True, self._get_info()

        self._prev_x = self.mario_x
        self.frame_count += 1
        if self.frame_count % 24 == 0:
            self.time = max(0, self.time - 1)
            if self.time <= 0:
                self._kill_mario()

        # Parse action
        move_right = action in (1, 2, 3, 4)
        move_left = action == 6
        jump = action in (2, 4, 5)
        run = action in (3, 4)
        max_speed = MARIO_MAX_RUN if run else MARIO_MAX_WALK

        # Horizontal movement
        if move_right:
            self.mario_vx += MARIO_ACCEL
            if self.mario_vx > max_speed:
                self.mario_vx = max_speed
            self.facing_right = True
        elif move_left:
            self.mario_vx -= MARIO_ACCEL
            if self.mario_vx < -max_speed:
                self.mario_vx = -max_speed
            self.facing_right = False
        else:
            if self.mario_vx > 0:
                self.mario_vx = max(0, self.mario_vx - MARIO_DECEL)
            elif self.mario_vx < 0:
                self.mario_vx = min(0, self.mario_vx + MARIO_DECEL)

        # Jump
        if jump and self.on_ground and not self.jumping:
            self.mario_vy = JUMP_VEL
            self.on_ground = False
            self.jumping = True
            self.jump_held = True
        elif jump and self.jumping:
            self.jump_held = True
        else:
            self.jump_held = False
            if self.jumping and self.mario_vy < JUMP_VEL_SHORT:
                self.mario_vy = max(self.mario_vy, JUMP_VEL_SHORT)

        # Gravity
        self.mario_vy += GRAVITY
        if self.mario_vy > MAX_FALL:
            self.mario_vy = MAX_FALL

        # Move X
        self.mario_x += self.mario_vx
        # Clamp left
        if self.mario_x < self.camera_x:
            self.mario_x = self.camera_x
            self.mario_vx = 0

        mario_rect = pygame.Rect(int(self.mario_x), int(self.mario_y), self.mario_w, self.mario_h)
        for r, c, t in self._collide_tile_rect(mario_rect):
            tile_rect = pygame.Rect(c * TILE, r * TILE, TILE, TILE)
            if mario_rect.colliderect(tile_rect):
                if self.mario_vx > 0:
                    self.mario_x = tile_rect.left - self.mario_w
                elif self.mario_vx < 0:
                    self.mario_x = tile_rect.right
                self.mario_vx = 0
                mario_rect.x = int(self.mario_x)

        # Move Y
        self.mario_y += self.mario_vy
        mario_rect = pygame.Rect(int(self.mario_x), int(self.mario_y), self.mario_w, self.mario_h)
        self.on_ground = False
        for r, c, t in self._collide_tile_rect(mario_rect):
            tile_rect = pygame.Rect(c * TILE, r * TILE, TILE, TILE)
            if mario_rect.colliderect(tile_rect):
                if self.mario_vy > 0:  # falling
                    self.mario_y = tile_rect.top - self.mario_h
                    self.mario_vy = 0
                    self.on_ground = True
                    self.jumping = False
                elif self.mario_vy < 0:  # hitting ceiling
                    self.mario_y = tile_rect.bottom
                    self.mario_vy = 0
                    # Hit question/brick block
                    if t == QUESTION:
                        self.level[r][c] = QUESTION_USED
                        self.coins += 1
                        self.score += 200
                    elif t == BRICK:
                        self.level[r][c] = EMPTY
                        self.score += 50
                mario_rect.y = int(self.mario_y)

        # Fell off screen
        if self.mario_y > 15 * TILE:
            self._kill_mario()

        # Flagpole check
        flag_col = None
        mario_rect = pygame.Rect(int(self.mario_x), int(self.mario_y), self.mario_w, self.mario_h)
        left_col = mario_rect.left // TILE
        right_col = mario_rect.right // TILE
        for c in range(left_col, right_col + 1):
            for r in range(15):
                if self._tile_at(c, r) in (FLAGPOLE, FLAG_TOP):
                    flag_col = c
                    break
            if flag_col is not None:
                break
        if flag_col is not None:
            self.won = True
            self.score += 1000

        # Update enemies
        self._update_enemies()

        # Camera
        screen_center = self.camera_x + LOGICAL_W * 0.4
        if self.mario_x > screen_center:
            self.camera_x = self.mario_x - LOGICAL_W * 0.4
        max_cam = self.level_w * TILE - LOGICAL_W
        self.camera_x = max(0, min(self.camera_x, max_cam))

        # Reward
        reward = 0.0
        dx = self.mario_x - self._prev_x
        reward += dx  # +1 per pixel right
        reward -= 0.1  # time pressure
        if self.dead:
            reward = -15.0
        if self.won:
            reward += 100.0

        done = self.dead or self.won
        return self._get_obs(), reward, done, self._get_info()

    def _kill_mario(self):
        self.dead = True
        self.death_timer = 60

    def _update_enemies(self):
        mario_rect = pygame.Rect(int(self.mario_x), int(self.mario_y), self.mario_w, self.mario_h)
        for e in self.enemies:
            if not e.alive:
                if e.squish_timer > 0:
                    e.squish_timer -= 1
                continue

            # Only update enemies near camera
            if abs(e.x - self.camera_x) > LOGICAL_W + 32:
                continue

            # Gravity
            e.vy += GRAVITY
            if e.vy > MAX_FALL:
                e.vy = MAX_FALL

            # Move X
            e.x += e.vx
            er = e.rect()
            for r, c, t in self._collide_tile_rect(er):
                tr = pygame.Rect(c * TILE, r * TILE, TILE, TILE)
                if er.colliderect(tr):
                    if e.vx > 0:
                        e.x = tr.left - e.w
                    elif e.vx < 0:
                        e.x = tr.right
                    e.vx = -e.vx

            # Move Y
            e.y += e.vy
            er = e.rect()
            for r, c, t in self._collide_tile_rect(er):
                tr = pygame.Rect(c * TILE, r * TILE, TILE, TILE)
                if er.colliderect(tr):
                    if e.vy > 0:
                        e.y = tr.top - e.h
                        e.vy = 0
                    elif e.vy < 0:
                        e.y = tr.bottom
                        e.vy = 0

            # Fell off
            if e.y > 15 * TILE:
                e.alive = False
                continue

            # Collision with Mario
            er = e.rect()
            if er.colliderect(mario_rect) and not self.dead:
                # Stomp if Mario is falling and his feet are in the top half of the enemy
                mario_bottom = self.mario_y + self.mario_h
                enemy_mid = e.y + e.h * 0.75
                if self.mario_vy >= 0 and mario_bottom <= enemy_mid:
                    e.alive = False
                    e.squish_timer = 15
                    self.mario_vy = JUMP_VEL * 0.5
                    self.score += 100
                else:
                    self._kill_mario()

    def _get_obs(self):
        self._render_frame()
        pixels = pygame.surfarray.array3d(self.logical_surface)
        # pygame gives (W, H, 3), transpose to (H, W, 3)
        pixels = np.transpose(pixels, (1, 0, 2))
        return pixels.astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "coins": self.coins,
            "time": self.time,
            "x_pos": self.mario_x,
            "dead": self.dead,
            "won": self.won,
        }

    def _render_frame(self):
        surf = self.logical_surface
        surf.fill(SKY_BLUE)
        cam = int(self.camera_x)

        # Draw tiles
        start_col = max(0, cam // TILE)
        end_col = min(self.level_w, (cam + LOGICAL_W) // TILE + 2)
        for r in range(15):
            for c in range(start_col, end_col):
                t = self.level[r][c]
                if t == EMPTY:
                    continue
                color = TILE_COLORS.get(t, WHITE)
                x = c * TILE - cam
                y = r * TILE
                pygame.draw.rect(surf, color, (x, y, TILE, TILE))
                # Draw outline for bricks and question blocks
                if t in (BRICK, QUESTION, QUESTION_USED):
                    pygame.draw.rect(surf, BLACK, (x, y, TILE, TILE), 1)
                # Draw ? symbol
                if t == QUESTION:
                    txt = self.font.render("?", True, WHITE)
                    surf.blit(txt, (x + 4, y + 2))

        # Draw enemies
        for e in self.enemies:
            if not e.alive:
                if e.squish_timer > 0:
                    # Draw squished
                    sx = int(e.x - cam)
                    sy = int(e.y + e.h - 6)
                    color = GOOMBA_BROWN if e.kind == "goomba" else KOOPA_GREEN
                    pygame.draw.rect(surf, color, (sx, sy, e.w, 6))
                continue
            if abs(e.x - cam) > LOGICAL_W + 16:
                continue
            sx = int(e.x - cam)
            sy = int(e.y)
            if e.kind == "goomba":
                pygame.draw.ellipse(surf, GOOMBA_BROWN, (sx, sy, e.w, e.h))
                # eyes
                pygame.draw.rect(surf, WHITE, (sx + 3, sy + 3, 3, 3))
                pygame.draw.rect(surf, WHITE, (sx + 8, sy + 3, 3, 3))
            else:
                pygame.draw.rect(surf, KOOPA_GREEN, (sx, sy, e.w, e.h))

        # Draw Mario
        if not self.dead:
            mx = int(self.mario_x - cam)
            my = int(self.mario_y)
            # Body
            pygame.draw.rect(surf, MARIO_RED, (mx, my + 4, self.mario_w, 12))
            # Head
            pygame.draw.rect(surf, MARIO_SKIN, (mx + 2, my, 10, 6))

        # HUD
        hud_y = 2
        score_txt = self.font.render(f"SCORE {self.score:06d}", True, WHITE)
        coin_txt = self.font.render(f"COINS {self.coins:02d}", True, WHITE)
        time_txt = self.font.render(f"TIME {self.time:03d}", True, WHITE)
        surf.blit(score_txt, (8, hud_y))
        surf.blit(coin_txt, (100, hud_y))
        surf.blit(time_txt, (200, hud_y))

    def render(self):
        if self.screen is None:
            return
        self._render_frame()
        scaled = pygame.transform.scale(self.logical_surface, (WINDOW_W, WINDOW_H))
        self.screen.blit(scaled, (0, 0))
        pygame.display.flip()
        self.clock.tick(FPS)


def play():
    """Play the game with keyboard controls."""
    pygame.init()
    game = MarioGame(render_mode=True)
    game.reset()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        keys = pygame.key.get_pressed()
        # Map keys to action
        right = keys[pygame.K_RIGHT] or keys[pygame.K_d]
        left = keys[pygame.K_LEFT] or keys[pygame.K_a]
        jump = keys[pygame.K_SPACE] or keys[pygame.K_UP] or keys[pygame.K_w]
        run = keys[pygame.K_LSHIFT] or keys[pygame.K_z]

        if right and jump and run:
            action = 4
        elif right and jump:
            action = 2
        elif right and run:
            action = 3
        elif right:
            action = 1
        elif jump:
            action = 5
        elif left:
            action = 6
        else:
            action = 0

        obs, reward, done, info = game.step(action)
        game.render()

        if done:
            pygame.time.wait(1500)
            game.reset()

    pygame.quit()


if __name__ == "__main__":
    play()
