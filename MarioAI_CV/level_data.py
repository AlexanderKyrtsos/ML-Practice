"""
Level data: tile constants, level builder, and save/load utilities.
"""

import json
import os

LEVELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "levels")
LEVEL_1_1_PATH = os.path.join(LEVELS_DIR, "world_1_1.json")

TILE = 16

# Tile types
EMPTY = 0
GROUND = 1
BRICK = 2
QUESTION = 3
QUESTION_USED = 4
PIPE_TL = 5
PIPE_TR = 6
PIPE_BL = 7
PIPE_BR = 8
BLOCK = 9
FLAGPOLE = 10
FLAG_TOP = 11

# Enemy spawn data for World 1-1: (x, y, kind)
ENEMIES_1_1 = [
    (22 * TILE, 12 * TILE, "goomba"),
    (40 * TILE, 12 * TILE, "goomba"),
    (51 * TILE, 12 * TILE, "goomba"),
    (52.5 * TILE, 12 * TILE, "goomba"),
    (80 * TILE, 4 * TILE, "goomba"),
    (82 * TILE, 4 * TILE, "goomba"),
    (97 * TILE, 12 * TILE, "goomba"),
    (98.5 * TILE, 12 * TILE, "goomba"),
    (107 * TILE, 12 * TILE, "goomba"),
    (108.5 * TILE, 12 * TILE, "goomba"),
    (114 * TILE, 12 * TILE, "goomba"),
    (115.5 * TILE, 12 * TILE, "goomba"),
    (124 * TILE, 12 * TILE, "goomba"),
    (125.5 * TILE, 12 * TILE, "goomba"),
    (128 * TILE, 12 * TILE, "goomba"),
    (129.5 * TILE, 12 * TILE, "goomba"),
    (107 * TILE, 11 * TILE, "koopa"),
]

MARIO_START_1_1 = (40, 192)


def build_level_1_1():
    """Build World 1-1 layout. 212 tiles wide x 15 tiles tall."""
    W = 212
    H = 15
    level = [[EMPTY] * W for _ in range(H)]

    def fill_ground(x_start, x_end):
        for x in range(x_start, x_end):
            level[13][x] = GROUND
            level[14][x] = GROUND

    fill_ground(0, 69)
    fill_ground(71, 86)
    fill_ground(89, 153)
    fill_ground(155, 212)

    level[9][16] = QUESTION
    level[9][20] = BRICK
    level[9][21] = QUESTION
    level[9][22] = BRICK
    level[9][23] = QUESTION
    level[9][24] = BRICK
    level[5][22] = QUESTION

    def place_pipe(x, top_row, height):
        for row in range(top_row, top_row + height):
            if row == top_row:
                level[row][x] = PIPE_TL
                level[row][x + 1] = PIPE_TR
            else:
                level[row][x] = PIPE_BL
                level[row][x + 1] = PIPE_BR

    place_pipe(28, 11, 2)
    place_pipe(38, 10, 3)
    place_pipe(46, 9, 4)
    place_pipe(57, 9, 4)
    place_pipe(163, 11, 2)
    place_pipe(179, 11, 2)

    level[9][77] = QUESTION
    level[5][78] = BRICK
    level[5][79] = BRICK
    level[5][80] = BRICK
    level[5][81] = BRICK
    level[5][82] = BRICK
    level[5][83] = BRICK
    level[5][84] = BRICK
    level[5][85] = BRICK

    level[9][80] = BRICK
    level[9][81] = QUESTION
    level[9][82] = QUESTION
    level[9][83] = BRICK

    level[9][91] = BRICK
    level[5][92] = BRICK
    level[5][93] = BRICK
    level[5][94] = BRICK
    level[9][94] = BRICK
    level[9][95] = BRICK

    level[9][100] = QUESTION
    level[9][101] = QUESTION
    level[5][101] = QUESTION
    level[9][106] = QUESTION
    level[9][109] = BRICK
    level[5][109] = BRICK

    for i in range(4):
        for r in range(12 - i, 13):
            level[r][134 + i] = BLOCK
    for i in range(4):
        for r in range(12 - (3 - i), 13):
            level[r][140 + i] = BLOCK
    for i in range(4):
        for r in range(12 - i, 13):
            level[r][148 + i] = BLOCK
    for i in range(4):
        for r in range(12 - (3 - i), 13):
            level[r][153 + i] = BLOCK
    for i in range(8):
        for r in range(12 - i, 13):
            level[r][198 - 7 + i] = BLOCK

    for r in range(3, 13):
        level[r][198] = FLAGPOLE
    level[2][198] = FLAG_TOP

    return level


def save_level(path, tiles, enemies, mario_start):
    """Save level to JSON file.

    Args:
        path: file path
        tiles: 2D list [row][col] of tile IDs
        enemies: list of (x, y, kind) tuples
        mario_start: (x, y) tuple
    """
    data = {
        "width": len(tiles[0]),
        "height": len(tiles),
        "tiles": tiles,
        "enemies": [{"x": x, "y": y, "kind": k} for x, y, k in enemies],
        "mario_start": list(mario_start),
    }
    with open(path, "w") as f:
        json.dump(data, f)


def load_level(path):
    """Load level from JSON file.

    Returns:
        (tiles, enemies, mario_start) where
        tiles is 2D list, enemies is list of (x, y, kind), mario_start is (x, y)
    """
    with open(path, "r") as f:
        data = json.load(f)
    tiles = data["tiles"]
    enemies = [(e["x"], e["y"], e["kind"]) for e in data["enemies"]]
    mario_start = tuple(data["mario_start"])
    return tiles, enemies, mario_start


def ensure_default_levels():
    """Create the default world_1_1.json if it doesn't exist yet."""
    os.makedirs(LEVELS_DIR, exist_ok=True)
    if not os.path.exists(LEVEL_1_1_PATH):
        save_level(LEVEL_1_1_PATH, build_level_1_1(), ENEMIES_1_1, MARIO_START_1_1)
