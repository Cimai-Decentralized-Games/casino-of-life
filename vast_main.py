import os
import json
import logging
import sys
from fastapi import FastAPI, Request, HTTPException
import retro
import gym
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from typing import Dict
import time

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/workspace/casino-of-life/vast_debug.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)
app = FastAPI()

# Algorithm mapping
ALGORITHMS = {
    'PPO': PPO,
    'A2C': A2C,
    'DQN': DQN
}

# Add a dictionary to store training status
training_sessions: Dict[str, dict] = {}

def create_env(game: str, state: str):
    """Create and wrap the environment"""
    env = retro.make(
        game=game,
        state=state,
        render_mode="rgb_array"
    )
    return DummyVecEnv([lambda: env])

@app.post("/train")
async def train(request: Request):
    print("\n=== TRAIN ENDPOINT HIT ===")
    logger.debug("Train endpoint accessed")
    
    try:
        data = await request.json()
        print(f"Received data: {json.dumps(data, indent=2)}")
        logger.debug(f"Parsed request data: {data}")
        
        # Extract parameters
        game = data.get("save_state", "MortalKombatII-Genesis")
        state = data.get("state", "Level1.LiuKangVsJax.state")
        algo_name = data.get("algorithm", "PPO")
        
        # Get training parameters
        training_params = data.get("training_params", {})
        learning_rate = training_params.get("learning_rate", 3e-4)
        batch_size = training_params.get("batch_size", 64)
        timesteps = training_params.get("timesteps", 1000)
        
        # Create environment
        logger.debug(f"Creating environment for {game} - {state}")
        env = create_env(game, state)
        
        # Initialize model with correct parameters
        if algo_name not in ALGORITHMS:
            raise ValueError(f"Algorithm {algo_name} not supported. Available: {list(ALGORITHMS.keys())}")
            
        logger.debug(f"Initializing {algo_name}")
        model = ALGORITHMS[algo_name](
            "CnnPolicy",
            env,
            learning_rate=learning_rate,
            batch_size=batch_size,
            verbose=1
        )
        
        # Update status before training
        session_id = request.headers.get('X-Session-ID')
        logger.debug(f"Training request for session {session_id}")
        
        if session_id:
            training_sessions[session_id] = {
                "status": "training",
                "progress": 0,
                "currentReward": 0,
                "episodeCount": 0
            }
        
        # Modify the training loop to update status
        total_timesteps = training_params.get("timesteps", 1000)
        episodes = 0
        
        def callback(locals, globals):
            nonlocal episodes
            if session_id:
                episodes += 1
                training_sessions[session_id].update({
                    "progress": int((locals["self"].num_timesteps / total_timesteps) * 100),
                    "currentReward": float(locals.get("rewards", [0])[0]),
                    "episodeCount": episodes
                })
            return True
        
        # Train with callback
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback
        )
        
        # Update final status
        if session_id:
            training_sessions[session_id].update({
                "status": "completed",
                "progress": 100
            })
        
        # Save model
        model_path = f"/workspace/casino-of-life/models/{game}_{state}_{algo_name}"
        model.save(model_path)
        logger.debug(f"Model saved to {model_path}")
        
        return {
            "status": "completed",
            "message": f"Training completed using {algo_name}",
            "model_path": model_path,
            "parameters": {
                "algorithm": algo_name,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "timesteps": timesteps
            }
        }
        
    except Exception as e:
        if session_id and session_id in training_sessions:
            training_sessions[session_id]["status"] = "failed"
        logger.exception("Error in train endpoint:")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/training-status")
async def get_training_status(request: Request):
    try:
        data = await request.json()
        session_id = request.headers.get('X-Session-ID')
        logger.debug(f"Status request for session {session_id}")
        
        if not session_id:
            raise HTTPException(status_code=400, detail="Session ID is required")
            
        # If session exists, return its status
        if session_id in training_sessions:
            return training_sessions[session_id]
        
        # If session doesn't exist, return default status
        return {
            "status": "unknown",
            "progress": 0,
            "currentReward": 0,
            "episodeCount": 0
        }
        
    except Exception as e:
        logger.exception("Error in training status endpoint:")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("Starting FastAPI server on port 5000...")
    logger.info("Initializing FastAPI server")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000, log_level="debug")