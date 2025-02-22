"""
Casino of Life - Main entry point and API server
"""
import os
import json
import logging
import asyncio
import uuid
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, WebSocket, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from g4f.client import AsyncClient
from g4f.Provider.Blackbox import Blackbox

# Import core components
from casino_of_life.game_environments.retro_env_loader import RetroEnv
from casino_of_life.agents.dynamic_agent import DynamicAgent
from casino_of_life.agents.caballo_loko import CaballoLoko
from casino_of_life.utils.config import CHAT_WS_URL, DEFAULT_GAME, DEFAULT_STATE
from casino_of_life.client_bridge.agent_factory import AgentFactory
from casino_of_life.client_bridge.chat_client import BaseChatClient, get_chat_client
from casino_of_life.client_bridge.reward_evaluators import (
    RewardEvaluatorManager,
    BasicRewardEvaluator,
    StageCompleteRewardEvaluator,
    MultiObjectiveRewardEvaluator
)
from casino_of_life.game_environments.game_intergrations import GameIntegrations
from casino_of_life.client_bridge.retro_api import RetroAPI

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Casino of Life API")
client = AsyncClient()

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

# Initialize RetroAPI
retro_api = RetroAPI(
    game=DEFAULT_GAME,
    state=DEFAULT_STATE,
    scenario=None,
    players=1
)

# Initialize DynamicAgent
training_params = {
    "learning_rate": 3e-4,
    "batch_size": 32,
    "timesteps": 100000
}

dynamic_agent = DynamicAgent(
    retro_api=retro_api,
    training_params=training_params,
    reward_evaluators={}
)

# API Routes

@app.post('/services.php')
async def initialize_service(request: Request):
    """Initialize service endpoint"""
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
    """Chat endpoint for agent interaction"""
    try:
        data = await request.json()
        message = data.get('message')
        session_id = request.headers.get('X-Session-ID')

        if not message:
            return JSONResponse({"error": "Message is required"}, status_code=400)

        if session_id not in sessions:
            caballo = CaballoLoko()
            sessions[session_id] = {
                'agent': caballo,
                'messages': [{"role": "system", "content": caballo.system_message}]
            }

        session = sessions[session_id]
        session['messages'].append({"role": "user", "content": message})

        try:
            response = await client.chat.completions.create(
                provider=Blackbox,
                model="gpt-4",
                messages=session['messages'],
                temperature=0.7,
                max_tokens=1000
            )

            if response and response.choices:
                assistant_response = response.choices[0].message.content
                session['messages'].append({
                    "role": "assistant",
                    "content": assistant_response
                })
                return JSONResponse({
                    "response": assistant_response,
                    "agent": session['agent'].name
                })
            else:
                return JSONResponse(
                    {"error": "No response from model"},
                    status_code=500
                )

        except Exception as e:
            logger.error(f"Model error: {str(e)}")
            return JSONResponse(
                {"error": f"Chat service error: {str(e)}"},
                status_code=500
            )

    except Exception as e:
        logger.error(f"Request error: {str(e)}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/train")
async def train(request: Request):
    """Training endpoint for agent training"""
    try:
        data = await request.json()
        session_id = str(uuid.uuid4())
        
        env = RetroEnv(
            game=data.get("game", DEFAULT_GAME),
            state=data.get("state", DEFAULT_STATE),
            players=1
        )
        
        agent = DynamicAgent(
            env=env,
            training_params={
                "learning_rate": data.get("learning_rate", 3e-4),
                "batch_size": data.get("batch_size", 32),
                "timesteps": data.get("timesteps", 100000)
            }
        )
        
        # Start training in background task
        training_sessions[session_id] = {
            "status": "running",
            "agent": agent,
            "params": data
        }
        
        return JSONResponse({
            "session_id": session_id,
            "status": "started",
            "message": "Training started"
        })
            
    except Exception as e:
        logger.exception("Error in train endpoint:")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/training-status")
async def get_training_status(request: Request):
    """Get training status endpoint"""
    try:
        data = await request.json()
        session_id = data.get("session_id")
        
        if session_id not in training_sessions:
            return JSONResponse({"error": "Session not found"}, status_code=404)
            
        session = training_sessions[session_id]
        return JSONResponse({
            "status": session["status"],
            "progress": session.get("progress", 0),
            "metrics": session.get("metrics", {})
        })
        
    except Exception as e:
        logger.exception("Error in training status endpoint:")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
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
            if data.get("type") == "training_update":
                await websocket.send_json({
                    "status": session["status"],
                    "progress": session.get("progress", 0),
                    "metrics": session.get("metrics", {})
                })
            else:
                await websocket.send_json({"message": "Unknown command"})
                
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=6789)
