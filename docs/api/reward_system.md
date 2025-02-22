# Reward System API Reference

The reward system provides a flexible framework for defining and combining different reward functions for agent training.

## BaseRewardEvaluator

The foundation class for all reward evaluators.
```python
from casino_of_life.reward_evaluators import BaseRewardEvaluator
class CustomRewardEvaluator(BaseRewardEvaluator):
def init(self):
super().init()
```

### Methods

#### evaluate(state: dict) -> float
Calculate reward based on game state.
```python
def evaluate(self, state: dict) -> float:
# Custom reward logic
return reward_value
```

#### reset() -> None
Reset the evaluator's internal state.
```python
evaluator.reset()
```

## MultiObjectiveRewardEvaluator

Combines multiple reward evaluators with weighted importance.
```python
from casino_of_life.reward_evaluators import (
MultiObjectiveRewardEvaluator,
HealthRewardEvaluator,
ProgressRewardEvaluator
)
evaluator = MultiObjectiveRewardEvaluator([
(HealthRewardEvaluator(), 1.0),
(ProgressRewardEvaluator(), 0.5)
])
```

### Methods

#### add_evaluator(evaluator: BaseRewardEvaluator, weight: float) -> None
Add a new reward evaluator with specified weight.
```python
evaluator.add_evaluator(
ProgressRewardEvaluator(),
weight=0.5
)
```

#### update_weights(weights: dict) -> None
Update weights for existing evaluators.
```python
evaluator.update_weights({
"health": 2.0,
"progress": 0.3
})
```

## Built-in Evaluators

### HealthRewardEvaluator

Rewards based on health differences and damage dealt/received.
```python
from casino_of_life.reward_evaluators import HealthRewardEvaluator
evaluator = HealthRewardEvaluator(
health_reward=1.0,
damage_penalty=-1.0,
)
```

### ProgressRewardEvaluator

Rewards based on stage progression and completion.
```python
from casino_of_life.reward_evaluators import ProgressRewardEvaluator
evaluator = ProgressRewardEvaluator(
stage_complete_reward=100.0,
progress_reward=0.1
)
```

### ComboRewardEvaluator

Rewards for executing specific move combinations.
```python
from casino_of_life.reward_evaluators import ComboRewardEvaluator
evaluator = ComboRewardEvaluator(
combo_rewards={
"fireball": 10.0,
"uppercut": 5.0,
"special_move": 15.0
}
)
```


## Strategy-Based Rewards

### AggressiveRewardEvaluator

Emphasizes offensive play and damage dealing.
```python
from casino_of_life.reward_evaluators import AggressiveRewardEvaluator
evaluator = AggressiveRewardEvaluator(
damage_dealt_multiplier=2.0,
distance_penalty=0.1
)
```


### DefensiveRewardEvaluator

Emphasizes health preservation and blocking.
```python
from casino_of_life.reward_evaluators import DefensiveRewardEvaluator
evaluator = DefensiveRewardEvaluator(
block_reward=5.0,
health_preserved_bonus=2.0
)
```


## Custom Reward Creation

### Example: Tournament Reward
```python
from casino_of_life.reward_evaluators import BaseRewardEvaluator
class TournamentRewardEvaluator(BaseRewardEvaluator):
def init(self, win_bonus=100.0, round_bonus=50.0):
super().init()
self.win_bonus = win_bonus
self.round_bonus = round_bonus
self.rounds_won = 0
def evaluate(self, state: dict) -> float:
reward = 0.0
# Round completion reward
if state.get("round_complete"):
if state.get("round_won"):
reward += self.round_bonus
self.rounds_won += 1
# Tournament win reward
if self.rounds_won >= 2:
reward += self.win_bonus
return reward
def reset(self):
self.rounds_won = 0
```

## Reward Scaling

### RewardScaler

Utility for scaling and normalizing rewards.
```python

from casino_of_life.reward_evaluators import RewardScaler
scaler = RewardScaler(
scale_factor=0.01,
clip_range=(-1.0, 1.0)
)
scaled_reward = scaler.scale(raw_reward)
```

## Reward History

### RewardTracker

Tracks and analyzes reward statistics during training.
```python
from casino_of_life.reward_evaluators import RewardTracker
tracker = RewardTracker()
tracker.add_reward(episode=1, reward=100.0)
stats = tracker.get_statistics()
Returns:
{
"mean_reward": 85.5,
"max_reward": 100.0,
"min_reward": 50.0,
"std_reward": 15.2
}
```