# casino_of_life/src/game_environments/game_integrations.py
import os
import json
import logging
from typing import Optional
from pathlib import Path
import retro
from casino_of_life.src.utils.config import SCRIPT_DIR

# Get the absolute path to the project root and data directory
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
PACKAGE_ROOT = PROJECT_ROOT / "casino_of_life"
DATA_DIR = PACKAGE_ROOT / "data" / "stable"

class GameIntegrations:
    """Manages game integrations for Stable Retro."""
    def __init__(self, custom_integration_path: Optional[str] = None):
        """
        Initializes the game integrations manager.
        Args:
            custom_integration_path: Custom path for integrations if provided.
        """
        self.custom_integration_path = custom_integration_path if custom_integration_path else str(DATA_DIR)
        self._load_custom_integrations()
        logging.info(f"Initialized GameIntegrations with path: {self.custom_integration_path}")

    def _load_custom_integrations(self):
        """Loads custom integrations from a directory."""
        try:
            retro.data.Integrations.add_custom_path(self.custom_integration_path)
            logging.info(f"Custom integrations loaded from: {self.custom_integration_path}")
        except Exception as e:
            logging.error(f"Failed to load custom integrations from path: {self.custom_integration_path}, with error: {e}")

    def get_available_games(self):
        """Gets a list of all available games."""
        try:
            return retro.data.list_games(inttype=retro.data.Integrations.ALL)
        except Exception as e:
            logging.error(f"Failed to get the available games, with error: {e}")
            return []

    def load_integration_data(self, game_name: str):
        """
        Loads data, scenario, and metadata for a game from the integration files.
        
        Args:
            game_name: The name of the game.
            
        Returns:
            A dictionary containing game data, scenario, metadata or None.
        """
        try:
            game_path = Path(self.custom_integration_path) / game_name
            if not game_path.exists():
                raise FileNotFoundError(f"Integration for '{game_name}' not found at path: '{game_path}'")

            data_path = game_path / "data.json"
            scenario_path = game_path / "scenario.json"
            metadata_path = game_path / "metadata.json"
            state_path = game_path / "Level1.LiuKangVsJax.state"

            result = {}
            
            # Load JSON files if they exist
            for path, key in [
                (data_path, "data"),
                (scenario_path, "scenario"),
                (metadata_path, "metadata")
            ]:
                if path.exists():
                    with open(path, "r") as f:
                        result[key] = json.load(f)
                else:
                    result[key] = None

            # Add state path if it exists
            result["state"] = str(state_path) if state_path.exists() else None

            return result

        except Exception as e:
            logging.error(f"Error loading integration data: {e}")
            return None

    def create_integration(self, game_name: str, game_rom_path: str, default_state=None, data=None, scenario=None, metadata=None):
        """
        Creates an integration in the custom integration directory.
        
        Args:
            game_name: Name of the game
            game_rom_path: Path to the game's ROM file
            default_state: Path to the default state file
            data: The `data.json` file content as a dict
            scenario: The `scenario.json` file content as a dict
            metadata: The `metadata.json` file content as a dict
        """
        try:
            integration_path = Path(self.custom_integration_path) / game_name
            integration_path.mkdir(parents=True, exist_ok=True)

            if game_rom_path:
                rom_dest = integration_path / Path(game_rom_path).name
                os.symlink(os.path.abspath(game_rom_path), str(rom_dest))

                sha_path = integration_path / "rom.sha"
                self._create_sha_file(game_rom_path, str(sha_path))

            # Create JSON files
            if data:
                self._create_json_file(data, str(integration_path / "data.json"))
            if scenario:
                self._create_json_file(scenario, str(integration_path / "scenario.json"))
            if metadata:
                if default_state:
                    metadata["default_state"] = default_state
                self._create_json_file(metadata, str(integration_path / "metadata.json"))

            return {"message": f"Game integration created at: {integration_path}"}
        except Exception as e:
            logging.error(f"Failed to create integration: {e}")
            return {"message": f"Failed to create game integration: {e}"}

    def _create_sha_file(self, file_path: str, output_path: str):
        """Creates a sha file for the game's rom."""
        import hashlib
        try:
            hasher = hashlib.sha1()
            with open(file_path, "rb") as f:
                while chunk := f.read(4096):
                    hasher.update(chunk)

            with open(output_path, "w") as f:
                f.write(hasher.hexdigest())
        except Exception as e:
            logging.error(f"Failed to create sha file: {e}")
            raise

    def _create_json_file(self, json_data, file_path):
        """Creates a json file based on a dict."""
        try:
            with open(file_path, "w") as f:
                json.dump(json_data, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to create json file: {e}")
            raise