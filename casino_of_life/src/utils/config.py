# casino-of-life/src/utils/config.py
import os
from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
PACKAGE_ROOT = PROJECT_ROOT / "casino_of_life"
SCRIPT_DIR = Path(__file__).parent.parent

# Data paths
DATA_DIR = PACKAGE_ROOT / "data" / "stable"
SCENARIOS_DIR = DATA_DIR
STATES_DIR = DATA_DIR

# API Configuration
CHAT_WS_URL = os.getenv("CHAT_WS_URL", "http://70.69.205.56:57258")
CHAT_API_KEY = os.getenv("CHAT_API_KEY", None)

# Default game settings
DEFAULT_GAME = "MortalKombatII-Genesis"
DEFAULT_STATE = "Level1.LiuKangVsJax"
