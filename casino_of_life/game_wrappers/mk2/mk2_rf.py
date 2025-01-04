# casino_of_life_retro/game_wrappers/mk2_rf.py

def init_function(env):
    # Initialize game state
    env.reset()
    initial_state = env.get_game_state()
    initial_state.p1_health = 100
    initial_state.p2_health = 100
    initial_state.p1_rounds_won = 0
    initial_state.p2_rounds_won = 0
    return initial_state

def reward_function(game_state):
    # Define the reward based on the game state
    reward = 0
    if game_state.p1_health > game_state.p2_health:
        reward += 1
    if game_state.p1_rounds_won > game_state.p2_rounds_won:
        reward += 10
    return reward

def done_function(game_state):
    # Define when an episode is done
    return game_state.p1_rounds_won == 2 or game_state.p2_rounds_won == 2

def register_functions(rf_name):
    # You can define multiple reward functions and select them based on rf_name
    return init_function, reward_function, done_function