"""
Simple pygame tile-based level editor for Mario levels.

Controls:
  Arrow keys: pan the view
  Left-click: paint selected item
  Right-click: erase / delete enemy
  B: switch to Blocks tab
  M: switch to Mobs tab
  E: eraser tool
  Click palette item at bottom to select
  S: save to JSON
  L: load from JSON
  Escape: quit
"""

import pygame
import sys
import os
from tkinter import Tk, filedialog

from level_data import (
    TILE, EMPTY, GROUND, BRICK, QUESTION, QUESTION_USED,
    PIPE_TL, PIPE_TR, PIPE_BL, PIPE_BR, BLOCK, FLAGPOLE, FLAG_TOP,
    save_level, load_level, ensure_default_levels, LEVEL_1_1_PATH,
)

# Display
SCREEN_W, SCREEN_H = 1024, 600
BAR_H = 50  # bottom palette bar height
TAB_H = 24  # tab button row height
FPS = 60

# Colors (same as game)
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
GOOMBA_BROWN = (170, 80, 30)
KOOPA_GREEN = (0, 168, 68)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
DARK_GRAY = (40, 40, 40)
MID_GRAY = (60, 60, 60)
HIGHLIGHT = (255, 255, 100)

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

# Block palette: (id, label, color)
BLOCK_PALETTE = [
    (GROUND, "Ground", GROUND_BROWN),
    (BRICK, "Brick", BRICK_BROWN),
    (QUESTION, "Question", QUESTION_YELLOW),
    (BLOCK, "Block", BLOCK_GRAY),
    (PIPE_TL, "Pipe", PIPE_GREEN),
    (FLAGPOLE, "Flagpole", FLAGPOLE_WHITE),
    (FLAG_TOP, "Flag Top", FLAG_GREEN),
    (QUESTION_USED, "Used ?", QUESTION_USED_BROWN),
]

# Enemy palette: (kind, label, color)
ENEMY_PALETTE = [
    ("goomba", "Goomba", GOOMBA_BROWN),
    ("koopa", "Koopa", KOOPA_GREEN),
]

TAB_BLOCKS = "blocks"
TAB_MOBS = "mobs"


def _ask_file(save=False):
    root = Tk()
    root.withdraw()
    if save:
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
    else:
        path = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
    root.destroy()
    return path if path else None


def run_editor():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Mario Level Editor")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("arial", 13)
    font_sm = pygame.font.SysFont("arial", 11)

    # Level state — load from world_1_1.json (created if missing)
    ensure_default_levels()
    current_path = LEVEL_1_1_PATH
    tiles, enemy_list, mario_start = load_level(current_path)
    enemies = [[x, y, k] for x, y, k in enemy_list]
    mario_start = list(mario_start)

    cam_x = 0
    cam_y = 0
    scroll_speed = 16

    # Selection state
    active_tab = TAB_BLOCKS
    selected_block = GROUND
    selected_enemy = "goomba"
    eraser_mode = False

    # Layout
    bar_top = SCREEN_H - BAR_H - TAB_H
    grid_bottom = bar_top

    running = True
    while running:
        mouse_buttons = pygame.mouse.get_pressed()
        mx, my = pygame.mouse.get_pos()
        mods = pygame.key.get_mods()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_b:
                    active_tab = TAB_BLOCKS
                    eraser_mode = False
                elif event.key == pygame.K_m:
                    active_tab = TAB_MOBS
                    eraser_mode = False
                elif event.key == pygame.K_e:
                    eraser_mode = not eraser_mode
                elif event.key == pygame.K_s:
                    if mods & pygame.KMOD_SHIFT:
                        # Save-as to new file
                        path = _ask_file(save=True)
                        if path:
                            current_path = path
                    else:
                        path = current_path
                    if path:
                        enemy_tuples = [(e[0], e[1], e[2]) for e in enemies]
                        save_level(path, tiles, enemy_tuples, tuple(mario_start))
                elif event.key == pygame.K_l:
                    path = _ask_file(save=False)
                    if path:
                        tiles, enemy_list, mario_start = load_level(path)
                        enemies = [[x, y, k] for x, y, k in enemy_list]
                        mario_start = list(mario_start)
                        current_path = path

            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Check tab clicks
                if bar_top <= my < bar_top + TAB_H:
                    tab_w = 80
                    if mx < tab_w:
                        active_tab = TAB_BLOCKS
                    elif mx < tab_w * 2:
                        active_tab = TAB_MOBS

                # Check palette clicks
                elif my >= bar_top + TAB_H and event.button == 1:
                    palette = BLOCK_PALETTE if active_tab == TAB_BLOCKS else ENEMY_PALETTE
                    pw = SCREEN_W // max(len(palette), 1)
                    idx = mx // pw
                    if 0 <= idx < len(palette):
                        eraser_mode = False
                        if active_tab == TAB_BLOCKS:
                            selected_block = palette[idx][0]
                        else:
                            selected_enemy = palette[idx][0]

                # Grid click
                elif my < grid_bottom:
                    grid_x = (mx + cam_x) // TILE
                    grid_y = (my + cam_y) // TILE
                    if eraser_mode:
                        if event.button == 1:
                            # Erase tile
                            if 0 <= grid_y < len(tiles) and 0 <= grid_x < len(tiles[0]):
                                tiles[grid_y][grid_x] = EMPTY
                            # Also delete any enemy in this cell
                            cx, cy = grid_x * TILE, grid_y * TILE
                            enemies = [e for e in enemies if not (abs(e[0] - cx) < TILE and abs(e[1] - cy) < TILE)]
                    elif active_tab == TAB_MOBS:
                        if event.button == 1:
                            enemies.append([grid_x * TILE, grid_y * TILE, selected_enemy])
                        elif event.button == 3:
                            cx, cy = grid_x * TILE, grid_y * TILE
                            enemies = [e for e in enemies if not (abs(e[0] - cx) < TILE and abs(e[1] - cy) < TILE)]
                    else:
                        if 0 <= grid_y < len(tiles) and 0 <= grid_x < len(tiles[0]):
                            if event.button == 1:
                                if selected_block == PIPE_TL:
                                    if grid_x + 1 < len(tiles[0]):
                                        tiles[grid_y][grid_x] = PIPE_TL
                                        tiles[grid_y][grid_x + 1] = PIPE_TR
                                else:
                                    tiles[grid_y][grid_x] = selected_block
                            elif event.button == 3:
                                tiles[grid_y][grid_x] = EMPTY

        # Continuous painting while mouse held
        if my < grid_bottom:
            grid_x = (mx + cam_x) // TILE
            grid_y = (my + cam_y) // TILE
            if 0 <= grid_y < len(tiles) and 0 <= grid_x < len(tiles[0]):
                if eraser_mode and mouse_buttons[0]:
                    tiles[grid_y][grid_x] = EMPTY
                elif active_tab == TAB_BLOCKS:
                    if mouse_buttons[0]:
                        if selected_block == PIPE_TL:
                            if grid_x + 1 < len(tiles[0]):
                                tiles[grid_y][grid_x] = PIPE_TL
                                tiles[grid_y][grid_x + 1] = PIPE_TR
                        else:
                            tiles[grid_y][grid_x] = selected_block
                    elif mouse_buttons[2]:
                        tiles[grid_y][grid_x] = EMPTY

        # Scroll
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            cam_x = max(0, cam_x - scroll_speed)
        if keys[pygame.K_RIGHT]:
            cam_x += scroll_speed
        if keys[pygame.K_UP]:
            cam_y = max(0, cam_y - scroll_speed)
        if keys[pygame.K_DOWN]:
            cam_y += scroll_speed

        # --- Draw ---
        screen.fill(DARK_GRAY)

        # Draw grid area
        grid_area = pygame.Rect(0, 0, SCREEN_W, grid_bottom)
        pygame.draw.rect(screen, SKY_BLUE, grid_area)

        start_col = cam_x // TILE
        end_col = (cam_x + SCREEN_W) // TILE + 1
        start_row = cam_y // TILE
        end_row = (cam_y + grid_bottom) // TILE + 1

        for r in range(max(0, start_row), min(len(tiles), end_row)):
            for c in range(max(0, start_col), min(len(tiles[0]), end_col)):
                t = tiles[r][c]
                if t == EMPTY:
                    continue
                sx = c * TILE - cam_x
                sy = r * TILE - cam_y
                color = TILE_COLORS.get(t, WHITE)
                pygame.draw.rect(screen, color, (sx, sy, TILE, TILE))
                pygame.draw.rect(screen, BLACK, (sx, sy, TILE, TILE), 1)

        # Draw enemies
        for e in enemies:
            sx = int(e[0]) - cam_x
            sy = int(e[1]) - cam_y
            color = GOOMBA_BROWN if e[2] == "goomba" else KOOPA_GREEN
            pygame.draw.rect(screen, color, (sx, sy, 14, 14))
            pygame.draw.rect(screen, WHITE, (sx, sy, 14, 14), 1)
            label = "G" if e[2] == "goomba" else "K"
            txt = font_sm.render(label, True, WHITE)
            screen.blit(txt, (sx + 2, sy))

        # Draw mario start
        msx = mario_start[0] - cam_x
        msy = mario_start[1] - cam_y
        pygame.draw.rect(screen, (228, 0, 0), (msx, msy, 14, 16))
        txt = font_sm.render("M", True, WHITE)
        screen.blit(txt, (msx + 2, msy))

        # Grid lines
        for c in range(start_col, min(len(tiles[0]) + 1, end_col + 1)):
            sx = c * TILE - cam_x
            pygame.draw.line(screen, (80, 80, 80), (sx, 0), (sx, grid_bottom), 1)
        for r in range(start_row, min(len(tiles) + 1, end_row + 1)):
            sy = r * TILE - cam_y
            pygame.draw.line(screen, (80, 80, 80), (0, sy), (SCREEN_W, sy), 1)

        # --- Bottom bar ---
        # Tab row
        pygame.draw.rect(screen, DARK_GRAY, (0, bar_top, SCREEN_W, TAB_H))
        tab_w = 80
        for i, (tab_id, tab_label, tab_key) in enumerate([
            (TAB_BLOCKS, "Blocks (B)", "b"),
            (TAB_MOBS, "Mobs (M)", "m"),
        ]):
            tx = i * tab_w
            is_active = (active_tab == tab_id)
            bg = MID_GRAY if is_active else DARK_GRAY
            pygame.draw.rect(screen, bg, (tx, bar_top, tab_w, TAB_H))
            pygame.draw.rect(screen, WHITE if is_active else (100, 100, 100), (tx, bar_top, tab_w, TAB_H), 1)
            txt = font_sm.render(tab_label, True, WHITE if is_active else (160, 160, 160))
            screen.blit(txt, (tx + 6, bar_top + 5))

        # Palette row
        palette_top = bar_top + TAB_H
        pygame.draw.rect(screen, DARK_GRAY, (0, palette_top, SCREEN_W, BAR_H))

        if active_tab == TAB_BLOCKS:
            palette = BLOCK_PALETTE
            current = selected_block
        else:
            palette = ENEMY_PALETTE
            current = selected_enemy

        pw = SCREEN_W // max(len(palette), 1)
        for i, entry in enumerate(palette):
            item_id, label, color = entry
            bx = i * pw
            # Color swatch
            pygame.draw.rect(screen, color, (bx + 4, palette_top + 4, pw - 8, BAR_H - 20))
            # Selection highlight
            if item_id == current:
                pygame.draw.rect(screen, HIGHLIGHT, (bx + 2, palette_top + 2, pw - 4, BAR_H - 16), 2)
            # Label
            txt = font_sm.render(label, True, WHITE)
            screen.blit(txt, (bx + 4, palette_top + BAR_H - 16))

        # HUD overlay (top-left)
        grid_x = (mx + cam_x) // TILE
        grid_y = (my + cam_y) // TILE
        if eraser_mode:
            sel_name = "Eraser"
        elif active_tab == TAB_BLOCKS:
            sel_name = next((l for tid, l, c in BLOCK_PALETTE if tid == selected_block), "?")
        else:
            sel_name = next((l for kid, l, c in ENEMY_PALETTE if kid == selected_enemy), "?")
        filename = os.path.basename(current_path)
        hud = f"({grid_x}, {grid_y})  Tool: {sel_name}  |  File: {filename}  |  S=Save  Shift+S=Save As  L=Load  E=Eraser"
        hud_surf = font.render(hud, True, WHITE)
        # Background behind HUD text for readability
        hud_bg = pygame.Surface((hud_surf.get_width() + 8, hud_surf.get_height() + 4), pygame.SRCALPHA)
        hud_bg.fill((0, 0, 0, 140))
        screen.blit(hud_bg, (4, 2))
        screen.blit(hud_surf, (8, 4))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    run_editor()
