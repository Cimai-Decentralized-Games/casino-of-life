# casino_of_life_retro/game_wrappers/mk2/mk2_const.py

class GameConsts:
    # Screen dimensions
    SCREEN_WIDTH = 320
    SCREEN_HEIGHT = 224

    # Action indices
    MOVE_LEFT = 0
    MOVE_RIGHT = 1
    MOVE_UP = 2
    MOVE_DOWN = 3
    PUNCH = 4
    KICK = 5
    BLOCK = 6
    SPECIAL = 7

    ACTIONS = [MOVE_LEFT, MOVE_RIGHT, MOVE_UP, MOVE_DOWN, PUNCH, KICK, BLOCK, SPECIAL]

    # Game state constants
    MAX_HEALTH = 100
    MAX_STAMINA = 100
    MAX_ROUNDS = 3
    ROUND_TIME = 60  # in seconds

    # Distance thresholds
    CLOSE_DISTANCE = 50
    MID_DISTANCE = 100
    FAR_DISTANCE = 150

    # Player states
    STATE_IDLE = 0
    STATE_WALKING = 1
    STATE_JUMPING = 2
    STATE_CROUCHING = 3
    STATE_ATTACKING = 4
    STATE_BLOCKING = 5
    STATE_STUNNED = 6

    # Observation space parameters
    NUM_PARAMS = 16

    # Normalization constants
    MAX_X_POSITION = SCREEN_WIDTH
    MAX_Y_POSITION = SCREEN_HEIGHT
    MAX_DISTANCE = ((SCREEN_WIDTH ** 2) + (SCREEN_HEIGHT ** 2)) ** 0.5
    MAX_VELOCITY = 10  # Adjust based on actual max velocity in the game
    MAX_STATE = 6  # Based on the number of player states
    MAX_ACTION = len(ACTIONS) - 1

    # Reward constants
    REWARD_WIN = 1.0
    REWARD_LOSE = -1.0
    REWARD_DRAW = 0.0
    REWARD_DAMAGE_DEALT = 0.01
    REWARD_DAMAGE_RECEIVED = -0.01