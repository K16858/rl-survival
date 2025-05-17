import os
import sys
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
import pygame

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from env.env import IslandEnvironment, TileType
from agent.ppo_agent import PPOAgent
from agent.train_ppo import preprocess_vision, preprocess_status
from agent.utils import plot_vision, plot_training_progress

def evaluate_ppo_agent(
    model_path: str,
    env_size: int = 200,
    n_episodes: int = 10,
    max_steps: int = 500,
    render: bool = True,
    fps: int = 30,
    output_dir: str = 'ppo_evaluation_results'
):
    """
    Evaluate a trained PPO agent and save results
    
    Args:
        model_path: Path to the model file
        env_size: Size of the environment
        n_episodes: Number of episodes to evaluate
        max_steps: Maximum steps per episode
        render: Whether to render the environment
        fps: Frame rate for rendering
        output_dir: Directory to save evaluation results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create environment
    env = IslandEnvironment(size=env_size, seed=42, tile_size=8)
    
    # Vision field shape
    vision_shape = (11, 11)
    
    # Get status size
    obs = env.reset()
    vision = obs['vision']
    status = preprocess_status(obs, obs['position'])
    status_size = len(status)
    
    # Number of actions
    n_actions = len(env.action_space)
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create agent
    agent = PPOAgent(
        vision_shape=vision_shape,
        status_size=status_size,
        n_actions=n_actions,
        device=device
    )
    
    # Load model
    agent.load(model_path)
    
    # Track metrics
    episode_scores = []
    episode_steps = []
    episode_tiles_visited = []
    episode_river_visits = []
    
    # Evaluate over multiple episodes
    for i_episode in range(1, n_episodes + 1):
        print(f"Starting evaluation episode {i_episode}/{n_episodes}")
        
        # Reset environment
        obs = env.reset()
        vision = preprocess_vision(obs['vision'])
        status = preprocess_status(obs, obs['position'])
        score = 0
        
        # Track tiles visited in this episode
        tiles_visited = set()
        river_visits = 0
        
        # Record trajectory
        trajectory = [env.agent_pos]
        
        # Start episode
        for t in range(1, max_steps + 1):
            # Select action
            action, _, _ = agent.act(vision, status)
            
            # Take action
            next_obs, reward, done, _ = env.step(action)
            
            # Add current position to trajectory
            trajectory.append(env.agent_pos)
            
            # Record metrics
            score += reward
            i, j = map(int, env.agent_pos)
            tiles_visited.add((i, j))
            if env.map[i, j] == TileType.RIVER:
                river_visits += 1
            
            # Preprocess next state
            next_vision = preprocess_vision(next_obs['vision'])
            next_status = preprocess_status(next_obs, next_obs['position'])
            
            # Update state
            vision = next_vision
            status = next_status
            
            # Render
            if render:
                env.render()
                time.sleep(1.0 / fps)
            
            # Save screenshot every 100 steps
            if t % 100 == 0 and render:
                # Save screenshot (if pygame is initialized)
                if env.is_initialized:
                    pygame_img_path = os.path.join(output_dir, f"episode_{i_episode}_step_{t}.png")
                    pygame.image.save(env.screen, pygame_img_path)
            
            # Print progress
            if t % 100 == 0:
                print(f"Step {t}/{max_steps}, Score: {score:.2f}")
            
            if done:
                break
        
        # Record episode results
        episode_scores.append(score)
        episode_steps.append(t)
        episode_tiles_visited.append(len(tiles_visited))
        episode_river_visits.append(river_visits)
        
        # Print episode summary
        print(f"Episode {i_episode} completed in {t} steps")
        print(f"Score: {score:.2f}")
        print(f"Unique tiles visited: {len(tiles_visited)}")
        print(f"River visits: {river_visits}")
        print("------------------------------")
        
        # Save agent's final vision
        vision_path = os.path.join(output_dir, f"episode_{i_episode}_final_vision.png")
        plot_vision(obs['vision'], save_path=vision_path)
        
        # Draw episode trajectory
        plt.figure(figsize=(10, 10))
        # First draw the map background
        map_img = np.zeros((env_size, env_size, 3), dtype=np.uint8)
        for i in range(env_size):
            for j in range(env_size):
                color = TileType.get_color(env.map[i, j])
                map_img[i, j] = color
        
        plt.imshow(map_img)
        
        # Draw trajectory
        traj_x = [pos[1] for pos in trajectory]
        traj_y = [pos[0] for pos in trajectory]
        plt.plot(traj_x, traj_y, 'r-', linewidth=2, alpha=0.7)
        plt.plot(traj_x[0], traj_y[0], 'go', markersize=8)  # Start position
        plt.plot(traj_x[-1], traj_y[-1], 'ro', markersize=8)  # End position
        
        plt.title(f"Episode {i_episode} Trajectory")
        plt.savefig(os.path.join(output_dir, f"episode_{i_episode}_trajectory.png"))
        plt.close()
    
    # Close environment
    env.close()
    
    # Calculate statistics
    avg_score = np.mean(episode_scores)
    avg_steps = np.mean(episode_steps)
    avg_tiles = np.mean(episode_tiles_visited)
    avg_river_visits = np.mean(episode_river_visits)
    
    # Print summary
    print("===== Evaluation Summary =====")
    print(f"Average Score: {avg_score:.2f}")
    print(f"Average Steps: {avg_steps:.1f}")
    print(f"Average Unique Tiles Visited: {avg_tiles:.1f}")
    print(f"Average River Visits: {avg_river_visits:.2f}")
    
    # Save summary to file
    with open(os.path.join(output_dir, "evaluation_summary.txt"), "w") as f:
        f.write("===== Evaluation Summary =====\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Environment Size: {env_size}\n")
        f.write(f"Episodes: {n_episodes}\n")
        f.write(f"Max Steps: {max_steps}\n\n")
        f.write(f"Average Score: {avg_score:.2f}\n")
        f.write(f"Average Steps: {avg_steps:.1f}\n")
        f.write(f"Average Unique Tiles Visited: {avg_tiles:.1f}\n")
        f.write(f"Average River Visits: {avg_river_visits:.2f}\n\n")
        f.write("Episode details:\n")
        for i in range(n_episodes):
            f.write(f"Episode {i+1}: Score={episode_scores[i]:.2f}, "
                    f"Steps={episode_steps[i]}, "
                    f"Tiles={episode_tiles_visited[i]}, "
                    f"River Visits={episode_river_visits[i]}\n")
    
    # Plot score distribution
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, n_episodes+1), episode_scores)
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Evaluation Scores by Episode')
    plt.savefig(os.path.join(output_dir, "score_distribution.png"))
    
    return avg_score, avg_steps, avg_tiles, avg_river_visits

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate a trained PPO agent')
    parser.add_argument('--model', type=str, required=True, 
                        help='Path to the model file')
    parser.add_argument('--env_size', type=int, default=200, 
                        help='Size of the environment')
    parser.add_argument('--episodes', type=int, default=5, 
                        help='Number of episodes to evaluate')
    parser.add_argument('--max_steps', type=int, default=500, 
                        help='Maximum steps per episode')
    parser.add_argument('--no_render', action='store_true', 
                        help='Disable rendering')
    parser.add_argument('--fps', type=int, default=30, 
                        help='Frame rate for rendering')
    parser.add_argument('--output', type=str, default='ppo_evaluation_results', 
                        help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    evaluate_ppo_agent(
        model_path=args.model,
        env_size=args.env_size,
        n_episodes=args.episodes,
        max_steps=args.max_steps,
        render=not args.no_render,
        fps=args.fps,
        output_dir=args.output
    )
