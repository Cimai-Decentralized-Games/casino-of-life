# casino_of_life_retro/client_bridge/reward_evaluators.py
import logging

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
    def __init__(self, stage_complete_reward = 100.0):
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
    multi_eval = MultiObjectiveRewardEvaluator(evaluators=[basic_eval, stage_eval])

    # Create manager
    reward_manager = RewardEvaluatorManager()
    reward_manager.register_evaluator("basic", basic_eval)
    reward_manager.register_evaluator("stage", stage_eval)
    reward_manager.register_evaluator("multi", multi_eval)

    # Example Usage
    prev_info = {"health": 100, "done": False}
    current_info = {"health": 90, "done": False}
    prev_obs = [1,2,3]
    current_obs = [4,5,6]
    action = [0, 1, 0, 0, 0, 1]
    reward1 = reward_manager.evaluate_reward("basic", prev_info, current_info, prev_obs, current_obs, action)
    print(f"Basic Reward: {reward1}") # output -10.0

    reward2 = reward_manager.evaluate_reward("stage", prev_info, current_info, prev_obs, current_obs, action)
    print(f"Stage Reward: {reward2}") # output -0.1

    current_info2 = {"health": 110, "done": True}
    reward3 = reward_manager.evaluate_reward("multi", prev_info, current_info2, prev_obs, current_obs, action)
    print(f"Multi Reward: {reward3}") # output 110.0