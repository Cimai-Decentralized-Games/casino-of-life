# casino_of_life_retro/client_bridge/reward_evaluators.py
import logging
import numpy as np

class BaseRewardEvaluator:
    """ Base class for all reward evaluators. """
    def __init__(self):
        pass

    def evaluate(self, prev_info, current_info, prev_obs, current_obs, action):
        """
        Abstract method for evaluating the reward.

        Args:
            prev_info: previous information from the step function
            current_info: current information from the step function
            prev_obs: previous state
            current_obs: current state
            action: action that was taken
        """
        raise NotImplementedError("Subclasses must implement 'evaluate'")


class BasicRewardEvaluator(BaseRewardEvaluator):
    """Basic reward evaluator that rewards health and penalizes damage."""

    def __init__(self, health_reward=1.0, damage_penalty=-1.0):
        super().__init__()
        self.health_reward = health_reward
        self.damage_penalty = damage_penalty

    def evaluate(self, prev_info, current_info, prev_obs, current_obs, action):
        """
        Evaluates the reward by checking health changes.

        Args:
            prev_info: previous information from the step function
            current_info: current information from the step function
            prev_obs: previous state
            current_obs: current state
            action: action that was taken
        """
        reward = 0.0
        if 'health' in current_info and 'health' in prev_info:
            health_change = current_info['health'] - prev_info['health']
            reward += health_change * self.health_reward if health_change > 0 else health_change * self.damage_penalty
        if reward == 0.0:
            reward -= 0.1
        return reward


class StageCompleteRewardEvaluator(BaseRewardEvaluator):
    """ Reward evaluator that rewards completing a stage. """
    def __init__(self, stage_complete_reward=100.0):
        super().__init__()
        self.stage_complete_reward = stage_complete_reward

    def evaluate(self, prev_info, current_info, prev_obs, current_obs, action):
        """ Evaluates the reward based on whether the stage is finished """
        reward = 0.0
        if 'done' in current_info and current_info['done']:
            reward += self.stage_complete_reward
        if reward == 0.0:
            reward -= 0.1
        return reward


class ProgressRewardEvaluator(BaseRewardEvaluator):
    """Evaluates rewards based on game progression metrics."""
    
    def __init__(self, position_reward=0.1, score_reward=0.01):
        super().__init__()
        self.position_reward = position_reward
        self.score_reward = score_reward
        
    def evaluate(self, prev_info, current_info, prev_obs, current_obs, action):
        reward = 0.0
        
        # Position-based progress (for platformers/fighting games)
        if 'x_pos' in current_info and 'x_pos' in prev_info:
            pos_change = current_info['x_pos'] - prev_info['x_pos']
            reward += pos_change * self.position_reward
            
        # Score-based progress
        if 'score' in current_info and 'score' in prev_info:
            score_change = current_info['score'] - prev_info['score']
            reward += score_change * self.score_reward
            
        if reward == 0.0:
            reward -= 0.1
        return reward


class ComboRewardEvaluator(BaseRewardEvaluator):
    """Rewards for executing specific move combinations."""
    
    def __init__(self, combo_rewards=None):
        super().__init__()
        self.combo_rewards = combo_rewards or {
            'special_move': 10.0,
            'combo': 5.0
        }
        self.action_buffer = []
        self.buffer_size = 10
        
    def evaluate(self, prev_info, current_info, prev_obs, current_obs, action):
        reward = 0.0
        
        # Update action buffer
        self.action_buffer.append(action)
        if len(self.action_buffer) > self.buffer_size:
            self.action_buffer.pop(0)
            
        # Check for special moves/combos in buffer
        if 'special_move' in current_info and current_info['special_move']:
            reward += self.combo_rewards.get('special_move', 10.0)
            
        if 'combo_count' in current_info and current_info.get('combo_count', 0) > 0:
            reward += self.combo_rewards.get('combo', 5.0) * current_info['combo_count']
            
        if reward == 0.0:
            reward -= 0.1
        return reward


class MultiObjectiveRewardEvaluator(BaseRewardEvaluator):
    """ Combines several reward evaluators. """
    def __init__(self, evaluators):
        super().__init__()
        self.evaluators = evaluators

    def evaluate(self, prev_info, current_info, prev_obs, current_obs, action):
        """
        Evaluates the reward by combining the evaluation of other evaluators.

        Args:
            prev_info: previous information from the step function
            current_info: current information from the step function
            prev_obs: previous state
            current_obs: current state
            action: action that was taken
        """
        total_reward = 0.0
        for evaluator in self.evaluators:
            total_reward += evaluator.evaluate(prev_info, current_info, prev_obs, current_obs, action)
        return total_reward


class TournamentRewardEvaluator(BaseRewardEvaluator):
    """Evaluates rewards in tournament-style gameplay."""
    
    def __init__(self, win_bonus=100.0, round_bonus=50.0):
        super().__init__()
        self.win_bonus = win_bonus
        self.round_bonus = round_bonus
        self.rounds_won = 0
        
    def evaluate(self, prev_info, current_info, prev_obs, current_obs, action):
        reward = 0.0
        
        # Round completion reward
        if current_info.get("round_complete", False):
            if current_info.get("round_won", False):
                reward += self.round_bonus
                self.rounds_won += 1
        
        # Tournament win reward
        if self.rounds_won >= 2:  # Best of 3
            reward += self.win_bonus
            self.rounds_won = 0  # Reset for next tournament
            
        if reward == 0.0:
            reward -= 0.1
        return reward


class RewardScaler:
    """Utility for scaling and normalizing rewards."""
    
    def __init__(self, scale_factor=0.01, clip_range=(-1.0, 1.0)):
        self.scale_factor = scale_factor
        self.clip_range = clip_range
        
    def scale(self, reward):
        """Scale and clip the reward."""
        scaled = reward * self.scale_factor
        return np.clip(scaled, self.clip_range[0], self.clip_range[1])


class RewardTracker:
    """Tracks and analyzes reward statistics during training."""
    
    def __init__(self):
        self.rewards = []
        self.episodes = []
        
    def add_reward(self, episode: int, reward: float):
        """Add a reward observation."""
        self.rewards.append(reward)
        self.episodes.append(episode)
        
    def get_statistics(self) -> dict:
        """Calculate reward statistics."""
        if not self.rewards:
            return {
                "mean_reward": 0.0,
                "max_reward": 0.0,
                "min_reward": 0.0,
                "std_reward": 0.0
            }
            
        rewards = np.array(self.rewards)
        return {
            "mean_reward": float(np.mean(rewards)),
            "max_reward": float(np.max(rewards)),
            "min_reward": float(np.min(rewards)),
            "std_reward": float(np.std(rewards))
        }
        
    def reset(self):
        """Clear tracked rewards."""
        self.rewards = []
        self.episodes = []


class RewardEvaluatorManager:
    """Manages the selection and application of reward evaluators."""

    def __init__(self, evaluators=None):
        """
        Args:
            evaluators: A dictionary of reward evaluator functions, where keys are names.
        """
        self.evaluators = evaluators if evaluators else {}

    def register_evaluator(self, name, evaluator):
        """
        Register a new reward evaluator.

        Args:
            name: Name of the evaluator.
            evaluator: The evaluator method.
        """
        if name in self.evaluators:
            logging.warning(f"Overwriting evaluator: '{name}'")
        self.evaluators[name] = evaluator

    def get_evaluator(self, name):
        """ Get an evaluator with a given name """
        if name not in self.evaluators:
            raise ValueError(f"Evaluator not found: '{name}'")
        return self.evaluators[name]

    def evaluate_reward(self, name, prev_info, current_info, prev_obs, current_obs, action):
        """
        Select and apply a reward evaluator.

        Args:
            name: The name of the reward evaluator.
            prev_info: previous information from the step function
            current_info: current information from the step function
            prev_obs: previous state
            current_obs: current state
            action: action that was taken

        Returns:
            The calculated reward.
        """
        if name not in self.evaluators:
            raise ValueError(f"Reward evaluator '{name}' not found.")
        evaluator = self.evaluators[name]
        try:
            reward = evaluator.evaluate(prev_info, current_info, prev_obs, current_obs, action)
            return reward
        except Exception as e:
            logging.error(f"Failed to evaluate reward, using default 0.0: {e}")
            return 0.0


# Example Usage
if __name__ == "__main__":
    # Create evaluators
    basic_eval = BasicRewardEvaluator()
    stage_eval = StageCompleteRewardEvaluator()
    progress_eval = ProgressRewardEvaluator()
    combo_eval = ComboRewardEvaluator()
    multi_eval = MultiObjectiveRewardEvaluator(
        evaluators=[basic_eval, stage_eval, progress_eval, combo_eval]
    )

    # Create manager
    reward_manager = RewardEvaluatorManager()
    reward_manager.register_evaluator("basic", basic_eval)
    reward_manager.register_evaluator("stage", stage_eval)
    reward_manager.register_evaluator("progress", progress_eval)
    reward_manager.register_evaluator("combo", combo_eval)
    reward_manager.register_evaluator("multi", multi_eval)

    # Example Usage
    prev_info = {
        "health": 100, 
        "done": False, 
        "x_pos": 100, 
        "score": 1000,
        "combo_count": 0
    }
    current_info = {
        "health": 90, 
        "done": False, 
        "x_pos": 120, 
        "score": 1200,
        "combo_count": 2,
        "special_move": True
    }
    prev_obs = [1,2,3]
    current_obs = [4,5,6]
    action = [0, 1, 0, 0, 0, 1]
    
    reward1 = reward_manager.evaluate_reward("basic", prev_info, current_info, prev_obs, current_obs, action)
    print(f"Basic Reward: {reward1}")
    
    reward2 = reward_manager.evaluate_reward("progress", prev_info, current_info, prev_obs, current_obs, action)
    print(f"Progress Reward: {reward2}")
    
    reward3 = reward_manager.evaluate_reward("combo", prev_info, current_info, prev_obs, current_obs, action)
    print(f"Combo Reward: {reward3}")
    
    reward4 = reward_manager.evaluate_reward("multi", prev_info, current_info, prev_obs, current_obs, action)
    print(f"Multi Reward: {reward4}")
    
    # Add tournament evaluator example
    tournament_eval = TournamentRewardEvaluator()
    reward_manager.register_evaluator("tournament", tournament_eval)
    
    # Add reward scaling example
    scaler = RewardScaler(scale_factor=0.01)
    
    # Add reward tracking example
    tracker = RewardTracker()
    
    # Example with tournament and tracking
    current_info["round_complete"] = True
    current_info["round_won"] = True
    
    reward = reward_manager.evaluate_reward("tournament", prev_info, current_info, prev_obs, current_obs, action)
    scaled_reward = scaler.scale(reward)
    tracker.add_reward(episode=1, reward=scaled_reward)
    
    print(f"Tournament Reward: {reward}")
    print(f"Scaled Reward: {scaled_reward}")
    print(f"Statistics: {tracker.get_statistics()}")