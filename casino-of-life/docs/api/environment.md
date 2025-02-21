# Environment API Reference

## RetroEnv

The main environment class for training and evaluation.
```python
from casino_of_life.environment import RetroEnv
env = RetroEnv(
game='MortalKombatII-Genesis',
state='tournament',
players=2
)
```
## Parameters

- `game`: The name of the game to load
- `state`: The initial state of the game
- `players`: The number of players in the game
- `frame_stack`: The number of frames to stack for the environment

## Methods
```python
step(action: np.ndarray) -> tuple
# Execute a single step in the environment
observation, reward, done, info = env.step(action)
reset() -> np.ndarray
# Reset the environment and return initial observation
observation = env.reset()
# Render the current environment state
env.render()
```
## State Management
```python
save_state(path: str) -> None
# Save the current environment state
env.save_state("states/mk2_tournament")
```
```python
load_state(path: str) -> None
# Load a previously saved state
env.load_state("states/mk2_tournament")
```

## Configuration

### Observation Space
- Type: Box(84, 84, 4)
- Range: [0, 255]
- Grayscale frames

### Action Space
- Type: MultiBinary(12)
- Buttons: ['B', 'A', 'MODE', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'C', 'Y', 'X', 'Z']

### Frame Processing
- Grayscale conversion
- Resolution: 84x84
- Frame stacking: 4 frames
- Frame skip: 4
