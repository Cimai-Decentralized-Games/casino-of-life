import os
import json
import logging
import asyncio
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, WebSocket, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from gpu_bridge import VastTrainingBridge
from g4f.client import Client
from g4f.Provider import OpenaiChat
from casino_of_life.src.trainers.rl_algorithms import rl_algorithm, PolicyType
from casino_of_life.src.trainers.interactive import Interactive
from casino_of_life.src.client_bridge.agent_factory import AgentFactory
from casino_of_life.src.client_bridge.chat_client import BaseChatClient, get_chat_client
from casino_of_life.src.client_bridge.reward_evaluators import (
    RewardEvaluatorManager, 
    BasicRewardEvaluator, 
    StageCompleteRewardEvaluator, 
    MultiObjectiveRewardEvaluator
)
from casino_of_life.src.game_environments.game_intergrations import GameIntegrations
from casino_of_life.agents.dynamic_agent import DynamicAgent
from casino_of_life.agents.agent_orchestrator import AgentOrchestrator
from casino_of_life.agents.character_package.character import Character
from casino_of_life.src.utils.config import CHAT_WS_URL, DEFAULT_GAME, DEFAULT_STATE
from casino_of_life.src.client_bridge.retro_api import RetroAPI
from casino_of_life.agents.custom_agent import BaseAgent
import sys
from casino_of_life.src.game_environments.retro_env_loader import RetroEnv
from casino_of_life.gpu_bridge import VastTrainingBridge
from casino_of_life.vast_config import VAST_INSTANCE

# Force debug logging to console
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)  # Force output to console
    ]
)

# Create logger for this file
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()
client = Client()


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global storage
sessions: Dict[str, Dict[str, Any]] = {}
training_sessions: Dict[str, Any] = {}

# Initialize components
game_integrations = GameIntegrations()
reward_manager = RewardEvaluatorManager()
agent_orchestrator = None

# Initialize dependencies first
retro_api = RetroAPI(
    game=DEFAULT_GAME,
    state=DEFAULT_STATE,
    scenario=None,
    players=1
)

# Create training parameters
training_params = {
    "learning_rate": 3e-4,
    "batch_size": 32,
    "timesteps": 100000
}

# Initialize DynamicAgent with RetroAPI, rl_algorithm and BaseAgent
dynamic_agent = DynamicAgent(
    retro_api=retro_api,
    rl_algorithm=rl_algorithm,
    training_params=training_params,
    reward_evaluators={}
)

agent_factory = None
chat_client = None

# Initialize the GPU bridge with your other global variables
vast_bridge = VastTrainingBridge()

def create_system_message(character):
    """Create the system message for the character."""
    try:
        name = character.name
        bio = character.bio
        examples = character.message_examples

        system_message = f"""You are {name}. {bio}

Here are some example messages of how you typically communicate:
"""
        if examples:
            for example in examples:
                system_message += f"\nExample: {example}\n"

        return {"role": "system", "content": system_message}
    except Exception as e:
        logger.error(f"Error creating system message: {e}", exc_info=True)
        return {"role": "system", "content": f"You are {character.name}."}

def setup_reward_evaluators(game: str, scenario: str = None) -> RewardEvaluatorManager:
    """Setup reward evaluators based on game and scenario."""
    manager = RewardEvaluatorManager()
    manager.add_evaluator(BasicRewardEvaluator())
    manager.add_evaluator(StageCompleteRewardEvaluator())
    
    if scenario:
        manager.add_evaluator(MultiObjectiveRewardEvaluator(scenario))
    
    return manager

def create_env(game: str, state: str, scenario: str = None):
    """Create a retro environment with game integrations."""
    try:
        env = game_integrations.create_environment(
            game=game,
            state=state,
            scenario=scenario
        )
        return env
    except Exception as e:
        logger.error(f"Error creating environment: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create environment: {str(e)}")

async def initialize_agent_factory():
    """Initialize the agent factory."""
    global agent_factory
    try:
        factory = AgentFactory(
            api_url=CHAT_WS_URL,
            dynamic_agent=dynamic_agent
        )
        agent_factory = await factory.initialize()
        return agent_factory
    except Exception as e:
        logger.error(f"Failed to initialize agent factory: {e}")
        raise

# Chat Service Routes
@app.post('/services.php')
async def initialize_service(request: Request):
    try:
        data = await request.json()
        service = data.get('service')
        
        if service == 'chat':
            return JSONResponse({
                "status": "success",
                "service": "chat",
                "initialized": True
            })
        else:
            return JSONResponse({"error": "Invalid service"}, status_code=400)
            
    except Exception as e:
        logger.error(f"Service initialization error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post('/chat')
async def chat(request: Request):
    try:
        data = await request.json()
        message = data.get('message')
        session_id = request.headers.get('X-Session-ID')

        if not message:
            return JSONResponse({"error": "Message is required"}, status_code=400)

        if session_id not in sessions:
            json_path = os.path.join(
                os.path.dirname(__file__),
                'casino_of_life/agents/character_package/character.json'
            )
            try:
                character = Character(json_path)
                sessions[session_id] = {
                    'character': character,
                    'messages': [create_system_message(character)]
                }
            except FileNotFoundError:
                return JSONResponse({"error": "Character configuration not found"}, status_code=404)

        session = sessions[session_id]
        session['messages'].append({"role": "user", "content": message})

        try:
            # Use the client API correctly
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=session['messages']
            )

            if response.choices[0].message.content:
                assistant_response = response.choices[0].message.content
                session['messages'].append({
                    "role": "assistant", 
                    "content": assistant_response
                })
                return JSONResponse({
                    "response": assistant_response,
                    "character": session['character'].name
                })
            else:
                return JSONResponse({"error": "No response generated"}, status_code=500)

        except Exception as e:
            logger.error(f"Chat error: {e}")
            return JSONResponse({"error": str(e)}, status_code=500)

    except Exception as e:
        logger.error(f"Request error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

# Training Service Routes
@app.post("/train")
async def train(request: Request):
    global agent_factory
    
    try:
        data = await request.json()
        logger.debug("=== TRAIN ENDPOINT ===")
        logger.debug(f"1. Raw data received: {data}")
        
        if not agent_factory:
            logger.debug("Initializing agent factory...")
            agent_factory = await initialize_agent_factory()
        
        message = data.get("message")
        if not message:
            logger.error("Missing message field in request")
            raise HTTPException(status_code=400, detail="Missing message field")
            
        logger.debug(f"2. Processing message: {message}")
            
        try:
            # Get training parameters from agent factory
            result = await agent_factory.create_agent(
                request_type=data.get("type"),
                message=message,
                game=DEFAULT_GAME,
                state=DEFAULT_STATE,
                scenario=None,
                players=1,
                action=None
            )
            
            # Start training on Vast.ai GPU
            training_result = await vast_bridge.start_training({
                "training_params": result.get("training_params", {}),
                "game": DEFAULT_GAME,
                "state": DEFAULT_STATE,
                "message": message
            })
            
            # Store training session
            session_id = training_result.get("session_id")
            training_sessions[session_id] = {
                "status": "running",
                "vast_instance": training_result.get("instance_id"),
                "params": result
            }
            
            logger.debug(f"3. Training started on GPU: {training_result}")
            return {
                "session_id": session_id,
                "status": "started",
                "message": "Training started on GPU"
            }
            
        except Exception as e:
            logger.exception("Error in create_agent:")
            raise
            
    except Exception as e:
        logger.exception("Error in train endpoint:")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/training-status")
async def get_training_status(request: Request):
    try:
        data = await request.json()
        session_id = data.get("session_id")
        
        if session_id not in training_sessions:
            return JSONResponse({"error": "Session not found"}, status_code=404)
            
        session = training_sessions[session_id]
        vast_instance_id = session.get("vast_instance")
        
        if vast_instance_id:
            # Get status from Vast.ai GPU
            status = vast_bridge.get_status(vast_instance_id)
            
            # Update session status
            session["status"] = status.get("status", "unknown")
            session["progress"] = status.get("progress", 0)
            session["metrics"] = status.get("metrics", {})
            
            return JSONResponse(session)
        else:
            return JSONResponse({"error": "No GPU instance found"}, status_code=404)
        
    except Exception as e:
        logger.exception("Error in training status endpoint:")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        session_id = None
        while True:
            data = await websocket.receive_json()
            
            if not session_id:
                session_id = data.get("session_id")
                if session_id not in training_sessions:
                    await websocket.send_json({"error": "Invalid session"})
                    continue
            
            session = training_sessions[session_id]
            vast_instance_id = session.get("vast_instance")
            
            if data.get("type") == "training_update" and vast_instance_id:
                status = vast_bridge.get_status(vast_instance_id)
                await websocket.send_json(status)
            else:
                await websocket.send_json({"message": "Unknown command"})
                
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=6000)

print("Loading RetroEnv from:", RetroEnv.__file__)
