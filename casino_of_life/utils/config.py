"""
Configuration settings for Casino of Life
"""

import os
from pathlib import Path

# Get package root directory
PACKAGE_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PACKAGE_ROOT / "data"
STABLE_DATA_DIR = DATA_DIR / "stable"
CONTRIB_DATA_DIR = DATA_DIR / "contrib"
EXPERIMENTAL_DATA_DIR = DATA_DIR / "experimental"

# Create directories if they don't exist
for dir in [DATA_DIR, STABLE_DATA_DIR, CONTRIB_DATA_DIR, EXPERIMENTAL_DATA_DIR]:
    dir.mkdir(parents=True, exist_ok=True)

# Environment variables
ENV_FILE = PACKAGE_ROOT / ".env"

# Model directories
MODELS_DIR = PACKAGE_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Scenarios directory
SCENARIOS_DIR = PACKAGE_ROOT / "scenarios"
SCENARIOS_DIR.mkdir(exist_ok=True)

# WebSocket configuration
CHAT_WS_URL = os.getenv("CHAT_WS_URL", "ws://localhost:6789/ws")

# Default game settings
DEFAULT_GAME = "MortalKombatII-Genesis"
DEFAULT_STATE = "tournament"

# API configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "6789"))

# Training defaults
DEFAULT_TRAINING_PARAMS = {
    "learning_rate": 3e-4,
    "batch_size": 32,
    "timesteps": 100000,
    "n_steps": 2048,
    "gamma": 0.99,
    "policy": "MlpPolicy"
}

__all__ = [
    'DATA_DIR',
    'STABLE_DATA_DIR',
    'CONTRIB_DATA_DIR',
    'EXPERIMENTAL_DATA_DIR',
    'MODELS_DIR',
    'SCENARIOS_DIR',
    'CHAT_WS_URL',
    'DEFAULT_GAME',
    'DEFAULT_STATE',
    'API_HOST',
    'API_PORT',
    'DEFAULT_TRAINING_PARAMS'
]
