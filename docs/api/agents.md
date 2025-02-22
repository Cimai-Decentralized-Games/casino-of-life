# Agents API Reference

## CaballoLoko

The main character agent that provides natural language interaction and training guidance.
```python
from casino_of_life.agents import CaballoLoko

caballo_loko = CaballoLoko()
```
### Methods

#### chat(message: str) -> str
Process a natural language message and return training guidance.
```python

response = caballo_loko.chat("Train Liu Kang to be aggressive with fireballs?")
```

#### post(message: str, context: dict) -> dict
Post a structured message with additional context.
```python
response = caballo.post(
message="Start training",
context={
"character": "liu-kang",
"strategy": "aggressive",
"policy": "PPO"
}
)
```

## DynamicAgent

The core training agent that implements reinforcement learning algorithms.
```python
from casino_of_life.agents import DynamicAgent
agent = DynamicAgent(
env=env,
policy='PPO',
learning_rate=0.0003
)
```

### Methods

#### train(timesteps: int, **kwargs) -> dict
Train the agent for a specified number of timesteps.
```python
agent.train(timesteps=1000000,
log_interval=1000,
save_interval=10000
)
```

#### evaluate(episodes: int) -> dict
Evaluate the agent's performance.
```python
metrics = agent.evaluate(episodes=10)
```

#### save_checkpoint(path: str) -> None
Save the agent's current state.
```python
agent.save_checkpoint("checkpoint.pth")
```

#### load_checkpoint(path: str) -> None
Load the agent's previous state.
```python
agent.load_checkpoint("checkpoint.pth")
```


## BaseAgent

The foundation class for all agents in the system.
```python
from casino_of_life.agents import BaseAgent
class CustomAgent(BaseAgent):
def init(self, env):
super().init(env)
```

### Properties

- `env`: The game environment
- `policy`: Current policy network
- `reward_evaluator`: Current reward evaluation system
- `training_state`: Current training state

### Methods

#### step(action: np.ndarray) -> tuple
Execute a single step in the environment.
```python
observation, reward, done, info = agent.step(action)
```

#### reset() -> np.ndarray
Reset the environment and return initial observation.
```python
observation = agent.reset()
```

