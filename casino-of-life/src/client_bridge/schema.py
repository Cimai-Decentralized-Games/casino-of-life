# casino_of_life_retro/client_bridge/schema.py
from pydantic import BaseModel, Field
from typing import Literal, Optional, Dict, Any

class TrainingRequest(BaseModel):
    """
    Schema for validating and structuring user input into a training request.
    """
    agent: str = Field(..., description="The name of the agent that will train the model for the game(e.g., ShaoKahn, Caballoloko, MasterPEPE, etc).")
    strategy: str = Field(
      ..., description="The training strategy for the model in the game. Can be one of: aggressive, defensive, balanced, or custom params."
    )
    game: str = Field(..., description="The name of the game (e.g., Super Mario Bros, Mortal Kombat).")
    state: Optional[str] = Field(
        "default", description="The state of the game to load (e.g., 'Level1', 'Stage2'). Defaults to 'default'."
    )
    scenario: Optional[str] = Field(
        None, description="The scenario configuration (if any)."
    )
    players: Optional[int] = Field(
        1, description="The number of players in the game. Defaults to 1."
    )
    total_timesteps: Optional[int] = Field(
        1000000, description="The total number of timesteps to train the agent for (default is 1,000,000)."
    )
    training_params: Optional[Dict[str, Any]] = Field(
        None, description="Custom training parameters."
    )

class StepRequest(BaseModel):
  """ Schema for stepping the game environment. """
  game: str = Field(..., description="The name of the game")
  state: str = Field(..., description="The state of the game to load")
  action: list[int] = Field(..., description="Action to send to the game")
  scenario: Optional[str] = Field(
      None, description="The scenario configuration (if any)."
  )
  players: Optional[int] = Field(
      1, description="The number of players in the game. Defaults to 1."
  )

class ResetRequest(BaseModel):
  """ Schema for resetting the game environment. """
  game: str = Field(..., description="The name of the game")
  state: str = Field(..., description="The state of the game to load")
  scenario: Optional[str] = Field(
      None, description="The scenario configuration (if any)."
  )
  players: Optional[int] = Field(
      1, description="The number of players in the game. Defaults to 1."
  )


# Example usage
if __name__ == "__main__":
    # Example input that would come from Chat's parser
    input_data = {
        "agent": "ShaoKahn",
        "strategy": "aggressive",
        "game": "MortalKombatII-Genesis",
        "total_timesteps": 1000000,
        "state": "default",
        "scenario": "default",
        "players": 1,
        "training_params": {"learning_rate": 0.001, "batch_size": 64}
    }

    # Validate and structure the input
    try:
        training_request = TrainingRequest(**input_data)
        print(training_request.json(indent=4))
    except Exception as e:
        print(f"Invalid input: {e}")