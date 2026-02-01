# MarioAI CV

A simplified Super Mario Bros World 1-1 clone built with pygame, with a Gym-compatible API for RL training and a built-in level editor.

## Requirements

- Python 3.10+
- pygame
- numpy

```
pip install pygame numpy
```

## Running the Game

```
python mario_game.py
```

### Controls

| Key | Action |
|---|---|
| Arrow Right / D | Move right |
| Arrow Left / A | Move left |
| Space / Up / W | Jump |
| Left Shift / Z | Run (hold with direction) |
| Escape | Quit |

The game loads the level from `levels/world_1_1.json`. Any edits made in the level editor are reflected the next time you start or reset the game.

## Level Editor

```
python level_editor.py
```

Opens `levels/world_1_1.json` by default. If the file doesn't exist, the default World 1-1 layout is generated automatically.

### Editor Controls

| Key | Action |
|---|---|
| B | Switch to Blocks tab |
| M | Switch to Mobs tab |
| E | Toggle eraser tool |
| Arrow keys | Pan the view |
| Left-click | Paint selected tile/mob (or erase in eraser mode) |
| Right-click | Erase tile / delete mob |
| S | Save to current file |
| Shift+S | Save as (file dialog) |
| L | Load a level (file dialog) |
| Escape | Quit |

You can also click items in the bottom palette bar to select them. The current tool and file name are shown in the top-left HUD.

### Block Types

Ground, Brick, Question, Block, Pipe, Flagpole, Flag Top, Used Question

### Mob Types

Goomba, Koopa

## Using as an RL Environment

```python
from mario_game import MarioGame

game = MarioGame(render_mode=False)
obs = game.reset()

while True:
    # action: 0=NOOP, 1=right, 2=right+jump, 3=right+run,
    #         4=right+run+jump, 5=jump, 6=left
    obs, reward, done, info = game.step(action)
    if done:
        obs = game.reset()
```

`step()` returns a `(240, 256, 3)` RGB numpy array as the observation. The `info` dict contains `score`, `coins`, `time`, `x_pos`, `dead`, and `won`.

To load a custom level:

```python
game = MarioGame(render_mode=False, level_path="levels/my_level.json")
```

## File Structure

| File | Purpose |
|---|---|
| `mario_game.py` | Game engine and playable entry point |
| `level_data.py` | Tile constants, level builder, JSON save/load |
| `level_editor.py` | Pygame level editor |
| `levels/world_1_1.json` | World 1-1 level data (auto-generated, editable) |
