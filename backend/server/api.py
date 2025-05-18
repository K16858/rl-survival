import os
import sys
import numpy as np
import torch
from fastapi import FastAPI, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Any, Optional, Tuple
import time
import json
from pathlib import Path

# Add path for module import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import custom modules
from train.ppo_agent import PPOAgent
from train.utils import create_experiment_dir
from env.env import IslandEnvironment, TileType

# State preprocessing functions (from train_ppo.py)
def preprocess_vision(vision):
    """Vision information preprocessing"""
    # Normalize tile information (0-6 -> 0-1)
    normalized = np.array(vision, dtype=np.float32) / 6.0
    # Add channel dimension
    return normalized.reshape(1, *normalized.shape)

def preprocess_status(obs, agent_pos):
    """Status information preprocessing"""
    # Extract and normalize necessary status information
    status_vector = np.array([
        agent_pos[0] / 500.0, agent_pos[1] / 500.0,  # Normalize position
        obs['hp'] / 100.0, 
        obs['stamina'] / 100.0,
        obs['hunger'] / 100.0,
        obs['hydration'] / 100.0,
        obs['temperature'],  # Already in 0-1
        obs['rest'] / 100.0,
        obs['stress'] / 100.0,
        float(obs['is_alive']),
        obs['days_survived'] / 30.0  # Normalize over 30 days
    ], dtype=np.float32)
    return status_vector

# Create FastAPI application
app = FastAPI(title="Survival AI API", description="API for survival simulation using reinforcement learning agent")

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Should be restricted to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models for requests/responses
class StateInput(BaseModel):
    vision: List[List[int]]
    position: Tuple[float, float]
    hp: float
    stamina: float
    hunger: float
    hydration: float 
    temperature: float
    rest: float
    stress: float
    is_alive: bool
    days_survived: float
    current_tile: int

class ActionResponse(BaseModel):
    action: int
    action_name: str
    confidence: float

class RewardInput(BaseModel):
    reward: float
    done: bool

class TrainingResponse(BaseModel):
    status: str
    message: str

class ModelInfo(BaseModel):
    model_path: str

# Global variables
agent = None
last_state = None
last_action = None
last_action_prob = None
last_value = None
model_dir = "models_api"
memory_buffer = []
MAX_MEMORY = 10000  # Maximum size of memory buffer
BATCH_SIZE = 64     # Batch size

# Action name mapping
ACTION_NAMES = [
    "Move Right",      # Right
    "Move Down-Right", # Down-Right
    "Move Down",       # Down
    "Move Down-Left",  # Down-Left
    "Move Left",       # Left
    "Move Up-Left",    # Up-Left
    "Move Up",         # Up
    "Move Up-Right",   # Up-Right
    "Use Item",        # Use inventory item
    "Drink Water",     # Drink water from river
    "Pick Up Item"     # Pick up nearest item
]

# Initialize agent
def init_agent(vision_shape=(11, 11), status_size=11, n_actions=11, model_path=None):
    global agent, model_dir
    
    # Create agent
    agent = PPOAgent(
        vision_shape=vision_shape,
        status_size=status_size,
        n_actions=n_actions,  # Updated to 11 actions
        gamma=0.99,
        gae_lambda=0.95,
        policy_clip=0.2,
        batch_size=BATCH_SIZE,
        n_epochs=10,
        lr=3e-4,
        seed=42
    )
    
    # Create model directory
    os.makedirs(model_dir, exist_ok=True)
    
    # Load existing model
    if model_path and os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        agent.load(model_path)
    else:
        print("Initializing new agent")

# Agent initialization at API startup
@app.on_event("startup")
async def startup_event():
    # Default model path
    default_model_path = os.path.join(model_dir, "best_model.pth")
    if os.path.exists(default_model_path):
        init_agent(model_path=default_model_path)
    else:
        init_agent()

# API usage flow:
# 1. Frontend sends state to backend via /api/get_action
# 2. Backend performs inference using the agent
# 3. Backend returns action to frontend
# 4. Frontend executes action and calculates reward
# 5. Frontend sends reward to backend via /api/learn
# 6. Backend performs learning with the received reward
# This loop continues for reinforcement learning

# Endpoint to get action from state
@app.post("/api/get_action", response_model=ActionResponse)
async def get_action(state: StateInput):
    global agent, last_state, last_action, last_action_prob, last_value
    
    # Initialize agent if not initialized
    if agent is None:
        init_agent()
    
    # Preprocess state
    state_dict = state.dict()
    vision = preprocess_vision(state_dict["vision"])
    status = preprocess_status(state_dict, state_dict["position"])
    
    # Let agent choose action
    action, log_prob, value = agent.act(vision, status)
    
    # Save current state and action
    last_state = {
        "vision": vision,
        "status": status
    }
    last_action = action
    last_action_prob = log_prob
    last_value = value
    
    # Convert to confidence (log_prob is log probability, so exp to get original probability)
    confidence = float(np.exp(log_prob))
    
    return ActionResponse(
        action=action, 
        action_name=ACTION_NAMES[action],
        confidence=confidence
    )

# Endpoint to receive reward and perform learning
@app.post("/api/learn", response_model=TrainingResponse)
async def learn(reward_input: RewardInput):
    global agent, last_state, last_action, last_action_prob, last_value, memory_buffer
    
    if agent is None:
        raise HTTPException(status_code=400, detail="Agent is not initialized")
    
    if last_state is None or last_action is None:
        raise HTTPException(status_code=400, detail="No previous state or action found")
    
    # Save transition to memory
    memory_buffer.append({
        "vision": last_state["vision"],
        "status": last_state["status"],
        "action": last_action,
        "log_prob": last_action_prob,
        "value": last_value,
        "reward": reward_input.reward,
        "done": reward_input.done
    })
    
    # Limit buffer size
    if len(memory_buffer) > MAX_MEMORY:
        memory_buffer = memory_buffer[-MAX_MEMORY:]
    
    # Learn if enough data is collected
    if len(memory_buffer) >= BATCH_SIZE:
        # Extract data from mini-batch and pass to agent's memory
        indices = np.random.choice(len(memory_buffer), BATCH_SIZE, replace=False)
        for i in indices:
            transition = memory_buffer[i]
            agent.memory.store(
                transition["vision"],
                transition["status"],
                transition["action"],
                transition["log_prob"],
                transition["value"],
                transition["reward"],
                transition["done"]
            )
        
        # Execute learning
        agent.learn()
        
        # Remove mini-batch from buffer
        memory_buffer = [t for i, t in enumerate(memory_buffer) if i not in indices]
    
    return TrainingResponse(
        status="success",
        message=f"Reward {reward_input.reward} processed. Memory buffer size: {len(memory_buffer)}"
    )

# Endpoint to save model
@app.post("/api/save_model", response_model=TrainingResponse)
async def save_model(model_info: Optional[ModelInfo] = None):
    global agent, model_dir
    
    if agent is None:
        raise HTTPException(status_code=400, detail="Agent is not initialized")
    
    # Determine save path
    if model_info and model_info.model_path:
        save_path = model_info.model_path
    else:
        # Path with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(model_dir, f"model_{timestamp}.pth")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save model
    agent.save(save_path)
    
    # Copy latest model as best_model
    best_model_path = os.path.join(model_dir, "best_model.pth")
    agent.save(best_model_path)
    
    return TrainingResponse(
        status="success",
        message=f"Model saved to {save_path}"
    )

# Endpoint to load model
@app.post("/api/load_model", response_model=TrainingResponse)
async def load_model(model_info: ModelInfo):
    global agent
    
    if not os.path.exists(model_info.model_path):
        raise HTTPException(status_code=404, detail=f"Model file not found: {model_info.model_path}")
    
    # Initialize agent if not initialized
    if agent is None:
        init_agent(model_path=model_info.model_path)
    else:
        agent.load(model_info.model_path)
    
    return TrainingResponse(
        status="success",
        message=f"Model loaded from {model_info.model_path}"
    )

# Endpoint to get a list of available models
@app.get("/api/available_models")
async def get_available_models():
    global model_dir
    
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Search for .pth files
    model_files = []
    for file in Path(model_dir).glob("**/*.pth"):
        model_files.append(str(file))
    
    return {"models": model_files}

# Endpoint to clear memory buffer
@app.post("/api/clear_memory", response_model=TrainingResponse)
async def clear_memory():
    global memory_buffer
    
    memory_buffer = []
    
    return TrainingResponse(
        status="success",
        message="Memory buffer cleared"
    )

# Main execution part
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
