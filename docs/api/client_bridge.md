# Client Bridge API Reference

The client bridge provides the interface between natural language commands, the training system, and the web interface.

## Parser

Converts natural language commands into structured training parameters.
```python
from casino_of_life.client_bridge import Parser
parser = Parser()
training_params = parser.parse_training_request("Train Liu Kang to be aggressive")
```

### Methods

#### parse_training_request(request: str) -> dict
Parses a natural language training request and returns a dictionary of training parameters.
```python
params = parser.parse_training_request("Train Liu Kang to be aggressive with fireballs")
Returns:
{
"character": "liu-kang",
"strategy": "aggressive",
"policy": "PPO",
"learning_rate": 0.0003,
"moves": ["fireball"],
"training_params": {
"timesteps": 100000,
"batch_size": 32
}
}
```

#### parse_strategy(message: str) -> str
Extract the training strategy from a message.
```python
strategy = parser.parse_strategy("Train Liu Kang to be aggressive with fireballs")
Returns: "aggressive"
```

## RewardEvaluatorManager

Manages and configures reward evaluators for different training scenarios.
```python
from casino_of_life.client_bridge import RewardEvaluatorManager
reward_manager = RewardEvaluatorManager()
```

### Methods

#### register_evaluator(name: str, evaluator: RewardEvaluator) -> None
Register a new reward evaluator.
```python
from casino_of_life.reward_evaluators import BasicRewardEvaluator
evaluator = BasicRewardEvaluator(
health_reward=1.0,
damage_penalty=-1.0
)
reward_manager.register_evaluator("tournament", evaluator)
```

#### get_evaluator(name: str) -> RewardEvaluator
Retrieve a registered reward evaluator.
```python
evaluator = reward_manager.get_evaluator("tournament")
```

#### modify_evaluator(name: str, **kwargs) -> None
Modify an existing evaluator's parameters.
```python
reward_manager.modify_evaluator(
"tournament",
health_reward=2.0,
damage_penalty=-0.5
)
```

## ActionMapper

Maps high-level actions to game-specific button combinations.
```python
from casino_of_life.client_bridge import ActionMapper
mapper = ActionMapper
```
### Methods

#### register_combo(name: str, buttons: list) -> None
Register a new button combination.
```python
mapper.register_combo(
"fireball",
["DOWN", "RIGHT", "B"]
)
```

#### get_action(combo_name: str) -> list
Get the button combination for an action.
```python
buttons = mapper.get_action("fireball")
```

#### execute_combo(env, combo_name: str) -> None
Execute a combo in the environment.
```python
mapper.execute_combo(env, "fireball")
```

## StateParser

Parses game state information from RAM and frame data.
```python
from casino_of_life.client_bridge import StateParser
parser = StateParser()
```

### Methods

#### parse_frame(observation: np.ndarray) -> dict
Extract game state from a frame.
```python
state = parser.parse_frame(observation)
Returns:
{
"player_position": [x, y],
"opponent_position": [x, y],
"health": {"p1": 100, "p2": 100},
"stage": "temple"
}
```

#### parse_ram(ram: np.ndarray) -> dict
Extract game state from RAM values.
```python
state = parser.parse_ram(ram)
Returns:
{
"health_p1": 100,
"health_p2": 100,
"stage_progress": 0.5,
"game_mode": "tournament"
}
```

## WebBridge

Handles communication between the training system and web interface.
```python
from casino_of_life.client_bridge import WebBridge
bridge = WebBridge()
```


### Methods

#### start_training_session(request: dict) -> str
Start a new training session from web request.
```python
session_id = bridge.start_training_session({
"message": "Train Liu Kang",
"strategy": "aggressive",
"policy": "PPO"
})
```

#### get_training_status(session_id: str) -> dict
Get current training status.
```python
status = bridge.get_training_status(session_id)
Returns:
{
"progress": 0.45,
"current_reward": 125.0,
"episodes_complete": 50,
"win_rate": 0.65
}
```


#### update_training(session_id: str, update: dict) -> None
Send updates to web client.
```python
bridge.update_training(
session_id,
{
"reward": 150.0,
"win": True,
"episode": 51
}
)
```


