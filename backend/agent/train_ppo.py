import os
import sys
import time
import numpy as np
import torch
from collections import deque
import matplotlib.pyplot as plt
import argparse

# Add parent directory to path for module import
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from env.env import IslandEnvironment, TileType
from agent.ppo_agent import PPOAgent
from agent.utils import plot_training_progress, create_experiment_dir, save_hyperparameters

def preprocess_vision(vision):
    """Normalize vision array for CNN input"""
    # Normalize to 0-1
    return vision.astype(np.float32) / max(1, vision.max())

def preprocess_status(obs, agent_pos):
    """Extract and normalize state features"""
    # Get tile type at current position
    current_tile = obs['current_tile']
    
    # One-hot encoding for tile type
    tile_type_onehot = np.zeros(7)  # Assume 7 tile types
    if current_tile < 7:
        tile_type_onehot[current_tile] = 1.0
    
    # Normalize position
    norm_pos = np.array(agent_pos) / 500.0  # Assume max map size 500
    
    # Concatenate features
    status = np.concatenate([norm_pos, tile_type_onehot])
    
    return status

def train_ppo(
    env_size=200,
    n_epochs=100,
    n_steps=2048,
    model_dir='models_ppo',
    log_interval=10,
    gamma=0.99,
    gae_lambda=0.95,
    policy_clip=0.2,
    batch_size=64,
    n_updates=10,
    lr=3e-4,
    seed=42,
    render=False,
    experiment_name=None
):
    """
    Train PPO agent in island environment
    
    Args:
        env_size: Size of the island map
        n_epochs: Number of training epochs
        n_steps: Steps per epoch
        model_dir: Directory to save models
        log_interval: Interval for logging progress
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        policy_clip: Policy clipping parameter
        batch_size: Batch size
        n_updates: Number of updates per batch
        lr: Learning rate
        seed: Random seed
        render: Whether to render environment
        experiment_name: Name of experiment (timestamp if None)
        
    Returns:
        scores: List of scores per epoch
    """
    # Create experiment directory
    if experiment_name:
        exp_dir = os.path.join(model_dir, experiment_name)
    else:
        exp_dir = create_experiment_dir(model_dir)
    
    os.makedirs(exp_dir, exist_ok=True)
    print(f"Experiment directory: {exp_dir}")
    
    # Save hyperparameters
    hyperparams = {
        'env_size': env_size,
        'n_epochs': n_epochs,
        'n_steps': n_steps,
        'gamma': gamma,
        'gae_lambda': gae_lambda,
        'policy_clip': policy_clip,
        'batch_size': batch_size,
        'n_updates': n_updates,
        'lr': lr,
        'seed': seed
    }
    save_hyperparameters(hyperparams, os.path.join(exp_dir, 'hyperparams.txt'))
    
    # Create environment
    env = IslandEnvironment(size=env_size, seed=seed)
    
    # Vision field shape
    vision_shape = (11, 11)  # From IslandEnvironment._get_observation
    
    # Get status size
    obs = env.reset()
    vision = obs['vision']
    status = preprocess_status(obs, obs['position'])
    status_size = len(status)
    
    # Number of actions
    n_actions = len(env.action_space)
    
    # Create agent
    agent = PPOAgent(
        vision_shape=vision_shape,
        status_size=status_size,
        n_actions=n_actions,
        gamma=gamma,
        gae_lambda=gae_lambda,
        policy_clip=policy_clip,
        batch_size=batch_size,
        n_epochs=n_updates,
        lr=lr,
        seed=seed
    )
    
    # Score records
    best_score = float('-inf')
    scores = []
    scores_window = deque(maxlen=100)  # Scores for last 100 epochs
    
    # Training loop
    for epoch in range(1, n_epochs+1):
        obs = env.reset()
        vision = preprocess_vision(obs['vision'])
        status = preprocess_status(obs, obs['position'])
        score = 0
        
        # Run steps for one epoch
        for step in range(n_steps):
            # Select action
            action, log_prob, value = agent.act(vision, status)
            
            # Take action
            next_obs, reward, done, _ = env.step(action)
            
            # Preprocess
            next_vision = preprocess_vision(next_obs['vision'])
            next_status = preprocess_status(next_obs, next_obs['position'])
            
            # Save data to memory
            agent.remember(vision, status, action, log_prob, value, reward, done)
            
            # Update state
            vision = next_vision
            status = next_status
            score += reward
            
            # Rendering
            if render:
                env.render()
                time.sleep(0.01)
            
            if step % 100 == 0:
                print(f"Epoch {epoch}/{n_epochs}, Step {step}/{n_steps}, Current score: {score:.2f}")
            
            if done:
                break
        
        # Learn after each epoch
        agent.learn()
        
        # Record score
        scores_window.append(score)
        scores.append(score)
        
        # Show progress
        if epoch % log_interval == 0:
            avg_score = np.mean(scores_window)
            print(f'Epoch {epoch}\tAverage Score: {avg_score:.2f}')
            
            # Plot training progress
            plot_training_progress(
                scores, 
                window_size=min(100, len(scores)),
                save_path=os.path.join(exp_dir, f'scores_epoch_{epoch}.png')
            )
        
        # Save best model
        if epoch % 10 == 0 or np.mean(scores_window) > best_score:
            if np.mean(scores_window) > best_score:
                best_score = np.mean(scores_window)
                agent.save(os.path.join(exp_dir, 'ppo_best.pth'))
                print(f"New best score: {best_score:.2f}, model saved.")
            
            # Periodic save
            if epoch % 10 == 0:
                agent.save(os.path.join(exp_dir, f'ppo_epoch_{epoch}.pth'))
    
    # Save final model
    agent.save(os.path.join(exp_dir, 'ppo_final.pth'))
    
    # Save final score plot
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(scores)), scores)
    plt.plot(np.arange(len(scores)), 
             [np.mean(scores[max(0, i-100):i+1]) for i in range(len(scores))],
             label='100-epoch moving average')
    plt.ylabel('Score')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(os.path.join(exp_dir, 'final_scores.png'))
    
    return scores

def test_ppo(
    model_path,
    env_size=200,
    n_episodes=5,
    max_steps=500,
    render=True,
    fps=10
):
    """
    Test trained PPO agent
    
    Args:
        model_path: Path to model to test
        env_size: Size of the island map
        n_episodes: Number of test episodes
        max_steps: Max steps per episode
        render: Whether to render environment
        fps: Rendering frame rate
    """
    # Create environment
    env = IslandEnvironment(size=env_size, seed=42)
    
    # Vision field shape
    vision_shape = (11, 11)
    
    # Get status size
    obs = env.reset()
    vision = obs['vision']
    status = preprocess_status(obs, obs['position'])
    status_size = len(status)
    
    # Number of actions
    n_actions = len(env.action_space)
    
    # Create agent
    agent = PPOAgent(
        vision_shape=vision_shape,
        status_size=status_size,
        n_actions=n_actions
    )
    
    # Load model
    agent.load(model_path)
    
    # Score records
    scores = []
    
    # Test loop
    for i_episode in range(1, n_episodes+1):
        obs = env.reset()
        vision = preprocess_vision(obs['vision'])
        status = preprocess_status(obs, obs['position'])
        score = 0
        
        for t in range(max_steps):
            # Rendering
            if render:
                env.render()
                time.sleep(1.0 / fps)
            
            # Select action
            action, _, _ = agent.act(vision, status)
            
            # Take action
            next_obs, reward, done, _ = env.step(action)
            
            # Preprocess
            next_vision = preprocess_vision(next_obs['vision'])
            next_status = preprocess_status(next_obs, next_obs['position'])
            
            # Update state
            vision = next_vision
            status = next_status
            score += reward
            
            if done:
                break
        
        scores.append(score)
        print(f'Episode {i_episode}\tScore: {score:.2f}')
    
    # Close environment
    env.close()
    
    # Show average score
    avg_score = np.mean(scores)
    print(f'Average score: {avg_score:.2f}')
    return avg_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train or test PPO agent in island environment')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                        help='Mode: train or test')
    parser.add_argument('--env_size', type=int, default=200,
                        help='Size of the island map')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--steps', type=int, default=2048,
                        help='Steps per epoch')
    parser.add_argument('--model_dir', type=str, default='models_ppo',
                        help='Directory to save/load models')
    parser.add_argument('--model_path', type=str, default='ppo_final.pth',
                        help='Model file name for testing')
    parser.add_argument('--render', action='store_true',
                        help='Render environment')
    parser.add_argument('--experiment', type=str, default=None,
                        help='Experiment name (directory)')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_ppo(
            env_size=args.env_size,
            n_epochs=args.epochs,
            n_steps=args.steps,
            model_dir=args.model_dir,
            render=args.render,
            experiment_name=args.experiment
        )
    else:
        model_path = os.path.join(args.model_dir, args.model_path)
        test_ppo(
            model_path=model_path,
            env_size=args.env_size,
            n_episodes=5,
            render=True
        )
