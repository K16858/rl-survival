import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
from typing import List, Dict, Tuple

def plot_vision(vision: np.ndarray, save_path: str = None):
    """
    Plot the agent's vision field
    
    Args:
        vision: Vision field array
        save_path: Path to save the plot (if None, just displays)
    """
    # Create a colormap for different tile types
    colors = [
        [0, 105/255, 148/255],  # OCEAN
        [194/255, 178/255, 128/255],  # BEACH
        [34/255, 139/255, 34/255],  # GRASS
        [64/255, 164/255, 223/255],  # RIVER
        [0, 100/255, 0],  # FOREST
        [139/255, 137/255, 137/255]  # MOUNTAIN
    ]
    cmap = ListedColormap(colors)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(vision, cmap=cmap, interpolation='nearest')
    plt.colorbar(ticks=range(len(colors)), label='Tile Type')
    plt.title('Agent Vision Field')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_value_map(model, env, status, device, save_path: str = None):
    """
    Plot a heatmap of Q-values over the entire map
    
    Args:
        model: The DQN model
        env: The environment
        status: Current status vector (excluding position)
        device: Device to use for inference
        save_path: Path to save the plot
    """
    # Create a grid of Q-values
    map_size = env.size
    q_values = np.zeros((map_size, map_size))
    
    # Current position for original status
    original_pos = env.agent_pos
    
    # Sample points on the map
    step_size = max(1, map_size // 50)  # Sample at most 50x50 points
    for i in range(0, map_size, step_size):
        for j in range(0, map_size, step_size):
            if env.map[i, j] != 0:  # Skip ocean
                # Set agent position
                env.agent_pos = (i + 0.5, j + 0.5)
                
                # Get observation
                obs = env._get_observation()
                vision = obs['vision']
                
                # Preprocess
                vision_tensor = torch.from_numpy(vision).unsqueeze(0).float().to(device)  # Add batch and channel dims
                status_tensor = torch.from_numpy(status).unsqueeze(0).float().to(device)
                
                # Get predicted value
                with torch.no_grad():
                    model.eval()
                    q_values_pred = model(vision_tensor, status_tensor)
                    max_q = q_values_pred.max().item()
                
                # Store the value
                q_values[i, j] = max_q
    
    # Reset agent position
    env.agent_pos = original_pos
    
    # Plot heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(q_values, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Max Q-value')
    plt.title('Value Function Heatmap')
    
    # Mark agent position
    plt.plot(int(original_pos[1]), int(original_pos[0]), 'ro', markersize=10)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_training_progress(scores: List[float], window_size: int = 100, save_path: str = None):
    """
    Plot training progress
    
    Args:
        scores: List of scores from each episode
        window_size: Window size for moving average
        save_path: Path to save the plot
    """
    # Calculate moving average
    moving_avg = []
    for i in range(len(scores)):
        if i < window_size:
            moving_avg.append(np.mean(scores[:i+1]))
        else:
            moving_avg.append(np.mean(scores[i-window_size+1:i+1]))
    
    plt.figure(figsize=(12, 6))
    plt.plot(scores, label='Score')
    plt.plot(moving_avg, label=f'Moving Avg (window={window_size})', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def save_hyperparameters(hyperparams: Dict, save_path: str):
    """
    Save hyperparameters to a text file
    
    Args:
        hyperparams: Dictionary of hyperparameters
        save_path: Path to save the file
    """
    with open(save_path, 'w') as f:
        f.write("Hyperparameters:\n")
        for key, value in hyperparams.items():
            f.write(f"{key}: {value}\n")

def create_experiment_dir(base_dir: str = 'experiments') -> str:
    """
    Create a new experiment directory with timestamp
    
    Args:
        base_dir: Base directory for experiments
        
    Returns:
        Path to the created directory
    """
    import datetime
    
    # Create base directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)
    
    # Create subdirectory with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_dir, f"exp_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    
    return exp_dir
