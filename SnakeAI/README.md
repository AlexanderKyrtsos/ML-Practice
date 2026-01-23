# Snake AI - Deep Q-Network Implementation

A complete Snake game with both human-playable GUI and headless AI training using Deep Q-Learning (DQN).

---

## Project Structure

```
SnakeAI/
├── snake_game.py      # Core game logic (no external dependencies except stdlib)
├── renderer.py        # Pygame-based GUI rendering
├── environment.py     # Gym-like wrapper for RL training
├── agent.py           # DQN neural network and training logic
├── train.py           # Training script with CLI
├── play.py            # Human play and AI demo script
├── requirements.txt   # Python dependencies
└── models/            # Saved model checkpoints (created during training)
```

---

## Core Game Logic (`snake_game.py`)

### Game State

The game maintains the following state:
- **Snake**: List of (x, y) coordinates, head first
- **Direction**: Current movement direction (UP=0, RIGHT=1, DOWN=2, LEFT=3)
- **Food**: (x, y) position of current food
- **Score**: Number of food items eaten
- **Game Over**: Boolean flag

### Coordinate System

```
(0,0) ────────────────► X (width)
  │
  │    Grid cells
  │
  ▼
  Y (height)
```

### Movement Rules

1. Snake moves one cell per step in current direction
2. Cannot reverse direction (would cause immediate self-collision)
3. Eating food: snake grows by 1, new food spawns
4. Death conditions: wall collision or self-collision
5. Timeout: game ends if no food eaten within `width * height * 2` steps

### Key Methods

| Method | Purpose |
|--------|---------|
| `reset()` | Initialize new game, return initial state |
| `step(direction)` | Advance game by one step, return (state, reward, done) |
| `get_observation()` | Generate 24-feature vector for neural network |
| `get_state()` | Return full GameState object for rendering |

---

## Observation Space (24 Features)

The neural network receives a 24-dimensional feature vector. This rich observation helps the agent understand not just immediate danger, but spatial awareness to avoid trapping itself.

### Feature Breakdown

| Index | Feature | Description | Range |
|-------|---------|-------------|-------|
| 0 | `danger_straight` | Immediate collision if going straight | 0 or 1 |
| 1 | `danger_right` | Immediate collision if turning right | 0 or 1 |
| 2 | `danger_left` | Immediate collision if turning left | 0 or 1 |
| 3 | `dist_straight` | Steps until collision going straight | 0.0 - 1.0 |
| 4 | `dist_right` | Steps until collision turning right | 0.0 - 1.0 |
| 5 | `dist_left` | Steps until collision turning left | 0.0 - 1.0 |
| 6-9 | `direction` | One-hot current direction (UP/RIGHT/DOWN/LEFT) | 0 or 1 |
| 10-13 | `food_direction` | Food relative to head (LEFT/RIGHT/UP/DOWN) | 0 or 1 |
| 14-17 | `tail_direction` | Tail relative to head (LEFT/RIGHT/UP/DOWN) | 0 or 1 |
| 18 | `space_straight` | Reachable cells if going straight | 0.0 - 1.0 |
| 19 | `space_right` | Reachable cells if turning right | 0.0 - 1.0 |
| 20 | `space_left` | Reachable cells if turning left | 0.0 - 1.0 |
| 21 | `body_dist_straight` | Distance to nearest body segment straight | 0.0 - 1.0 |
| 22 | `body_dist_right` | Distance to nearest body segment right | 0.0 - 1.0 |
| 23 | `body_dist_left` | Distance to nearest body segment left | 0.0 - 1.0 |

### Why These Features Matter

**Immediate Danger (0-2)**: Prevents obvious deaths.

**Distance to Danger (3-5)**: Helps plan a few steps ahead.

**Available Space (18-20)**: **Critical for avoiding self-trapping.** Uses flood-fill to count reachable cells. If going straight leads to 5 cells but left leads to 50, the agent learns to prefer open space.

**Body Proximity (21-23)**: Helps avoid getting close to own body.

**Tail Direction (14-17)**: Helps the snake understand its overall body position to avoid circling back.

---

## Action Space

The agent chooses from 3 relative actions:

| Action | Meaning |
|--------|---------|
| 0 | Continue straight |
| 1 | Turn right (relative to current direction) |
| 2 | Turn left (relative to current direction) |

Using relative actions (instead of absolute UP/DOWN/LEFT/RIGHT) simplifies learning since the agent doesn't need to learn separate policies for each direction.

---

## Reward Structure

| Event | Reward | Purpose |
|-------|--------|---------|
| Eat food | +10.0 | Primary objective |
| Move toward food | +0.1 | Encourage progress |
| Move away from food | -0.1 | Discourage wandering |
| Die (wall/self) | -10.0 | Punish death |
| Timeout | -5.0 | Punish stalling |

### Reward Shaping Rationale

- Large rewards for food (+10) and death (-10) establish clear objectives
- Small movement rewards (±0.1) guide exploration without overwhelming main objectives
- Timeout penalty prevents infinite loops where snake just survives without eating

---

## Neural Network Architecture (`agent.py`)

### DQN Structure

```
Input (24 features)
       │
       ▼
┌─────────────────┐
│ Linear(24, 256) │
│      ReLU       │
├─────────────────┤
│ Linear(256, 256)│
│      ReLU       │
├─────────────────┤
│ Linear(256, 3)  │  ◄── Q-values for each action
└─────────────────┘
       │
       ▼
Output (3 Q-values)
```

### Key Components

**Policy Network**: Makes action decisions, updated every step.

**Target Network**: Provides stable Q-value targets, updated every 100 steps.

**Replay Buffer**: Stores (state, action, reward, next_state, done) transitions. Samples random batches to break correlation in training data.

### Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_size` | 256 | Neurons per hidden layer |
| `learning_rate` | 0.001 | Adam optimizer LR |
| `gamma` | 0.99 | Discount factor for future rewards |
| `epsilon_start` | 1.0 | Initial exploration rate |
| `epsilon_end` | 0.01 | Minimum exploration rate |
| `epsilon_decay` | 0.995 | Epsilon multiplier per episode |
| `batch_size` | 64 | Training batch size |
| `buffer_size` | 100,000 | Replay buffer capacity |
| `target_update_freq` | 100 | Steps between target network updates |

---

## Training Process (`train.py`)

### Algorithm: Deep Q-Learning

```
For each episode:
    state = env.reset()

    While not done:
        1. Select action (ε-greedy):
           - Random action with probability ε
           - Best Q-value action with probability 1-ε

        2. Execute action, observe (next_state, reward, done)

        3. Store transition in replay buffer

        4. Sample random batch from buffer

        5. Compute target Q-values:
           target = reward + γ * max(Q_target(next_state)) * (1 - done)

        6. Update policy network:
           loss = MSE(Q_policy(state, action), target)
           backpropagate

        7. Periodically update target network

    Decay epsilon
```

### Epsilon Decay Schedule

```
Episode:    1      100     200     500     1000
Epsilon:   1.0    0.60    0.36    0.08    0.01
```

The agent starts with 100% random exploration and gradually shifts to exploiting learned knowledge.

### Training Flow

```
┌─────────────────────────────────────────────────────────────┐
│                      Training Loop                          │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐  │
│  │ Select  │───►│ Execute │───►│ Store   │───►│ Sample  │  │
│  │ Action  │    │ Step    │    │ Buffer  │    │ Batch   │  │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘  │
│       ▲                                            │        │
│       │                                            ▼        │
│  ┌─────────┐                              ┌─────────────┐   │
│  │ Update  │◄─────────────────────────────│   Compute   │   │
│  │ Target  │  (every 100 steps)           │    Loss     │   │
│  └─────────┘                              └─────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## Environment Wrapper (`environment.py`)

### SnakeEnv

Provides a clean interface for training:

```python
env = SnakeEnv(width=20, height=20)

state = env.reset()           # Returns numpy array (24,)
state, reward, done, info = env.step(action)  # action in {0, 1, 2}
```

### VectorizedSnakeEnv

Runs multiple games in parallel for faster training:

```python
vec_env = VectorizedSnakeEnv(n_envs=4)
states = vec_env.reset()      # Returns (4, 24) array
states, rewards, dones, infos = vec_env.step(actions)
```

---

## Renderer (`renderer.py`)

### SnakeRenderer

Pygame-based visualization with:
- Grid display with snake and food
- Score bar at top
- Snake head with eyes (indicates direction)
- Game over message
- Support for displaying training stats

### HeadlessRenderer

No-op renderer for fast headless training. Same interface, does nothing.

---

## Usage

### Installation

```bash
cd SnakeAI
pip install -r requirements.txt
```

### Play as Human

```bash
python play.py human
python play.py human --grid-size 15 --fps 12
```

**Controls:**
- Arrow keys or WASD: Move
- R: Restart
- Q/Escape: Quit

### Train AI (Headless)

```bash
# Basic training
python train.py --episodes 3000

# With visualization every 100 episodes
python train.py --episodes 3000 --visualize --viz-freq 100

# Resume training
python train.py --episodes 2000 --resume models/snake_dqn.pth

# Custom hyperparameters
python train.py --episodes 5000 --lr 0.0005 --hidden-size 512
```

### Watch AI Play

```bash
python play.py ai models/snake_dqn.pth
python play.py ai models/snake_dqn.pth --fps 15 --games 10
```

### Benchmark AI (Fast, No Rendering)

```bash
python play.py benchmark models/snake_dqn.pth --games 100
```

---

## Training Tips

### Recommended Settings for Good Results

```bash
python train.py --episodes 5000 --grid-size 10 --epsilon-decay 0.997
```

Smaller grid (10x10) trains faster and still demonstrates learning well.

### Signs of Good Training

- Average score steadily increases over episodes
- Snake consistently finds food in early game
- Snake avoids obvious traps
- Longer survival times

### Common Issues

| Problem | Likely Cause | Solution |
|---------|--------------|----------|
| Score stuck at 0-2 | Insufficient training | Train more episodes |
| Snake circles endlessly | Timeout too high | Reduce `max_steps_without_food` |
| Erratic behavior | Epsilon too high | Let training continue (epsilon decays) |
| Traps itself | Needs space awareness | Already fixed in v2 observations |

---

## File Dependencies

```
snake_game.py      ◄── No dependencies (pure Python)
       │
       ▼
environment.py     ◄── Imports snake_game
       │
       ▼
agent.py           ◄── Imports torch, numpy
       │
       ▼
train.py           ◄── Imports environment, agent, (optional: renderer)
       │
renderer.py        ◄── Imports pygame, snake_game
       │
       ▼
play.py            ◄── Imports all above
```

---

## Model Persistence

Models are saved as PyTorch checkpoint files containing:
- Policy network weights
- Target network weights
- Optimizer state
- Epsilon value
- Training progress (steps, episodes)

```python
# Save
agent.save("models/snake_dqn.pth")

# Load
agent = DQNAgent.from_file("models/snake_dqn.pth")
```

---

## Extending the Project

### Ideas for Improvement

1. **CNN-based observations**: Feed raw grid as image instead of handcrafted features
2. **Double DQN**: Reduce overestimation bias in Q-values
3. **Prioritized Experience Replay**: Sample important transitions more often
4. **Dueling DQN**: Separate state value and action advantage streams
5. **Multi-agent**: Multiple snakes competing for food
6. **Curriculum learning**: Start with small grid, gradually increase size
