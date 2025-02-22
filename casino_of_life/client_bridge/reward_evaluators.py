"""
Reward Evaluators Module - Handles different reward evaluation strategies
"""

from typing import Dict, Any, Optional

class RewardEvaluator:
    """Base class for reward evaluators"""
    def evaluate(self, info: Dict[str, Any]) -> float:
        raise NotImplementedError

class BasicRewardEvaluator(RewardEvaluator):
    """Basic reward evaluator focusing on damage output"""
    def evaluate(self, info: Dict[str, Any]) -> float:
        damage_dealt = info.get('damage_dealt', 0)
        damage_taken = info.get('damage_taken', 0)
        return damage_dealt - damage_taken

class StageCompleteRewardEvaluator(RewardEvaluator):
    """Reward evaluator focusing on stage completion"""
    def evaluate(self, info: Dict[str, Any]) -> float:
        stage_complete = info.get('stage_complete', False)
        return 100.0 if stage_complete else 0.0

class MultiObjectiveRewardEvaluator(RewardEvaluator):
    """Combines multiple reward objectives"""
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or {
            'damage': 1.0,
            'stage': 0.5,
            'health': 0.3
        }

    def evaluate(self, info: Dict[str, Any]) -> float:
        total_reward = 0.0
        if 'damage_dealt' in info:
            total_reward += self.weights['damage'] * (info['damage_dealt'] - info.get('damage_taken', 0))
        if 'stage_complete' in info:
            total_reward += self.weights['stage'] * (100.0 if info['stage_complete'] else 0.0)
        if 'health' in info:
            total_reward += self.weights['health'] * info['health']
        return total_reward

class RewardEvaluatorManager:
    """Manages different reward evaluators"""
    def __init__(self):
        self.evaluators = {
            'basic': BasicRewardEvaluator(),
            'stage': StageCompleteRewardEvaluator(),
            'multi': MultiObjectiveRewardEvaluator()
        }

    def add_evaluator(self, evaluator: RewardEvaluator, name: Optional[str] = None):
        """Add a new evaluator"""
        if name is None:
            name = evaluator.__class__.__name__
        self.evaluators[name] = evaluator

    def get_evaluator(self, name: str) -> Optional[RewardEvaluator]:
        """Get an evaluator by name"""
        return self.evaluators.get(name)

    def evaluate(self, name: str, info: Dict[str, Any]) -> float:
        """Evaluate using a specific evaluator"""
        evaluator = self.get_evaluator(name)
        if evaluator is None:
            raise ValueError(f"No evaluator found with name: {name}")
        return evaluator.evaluate(info)
