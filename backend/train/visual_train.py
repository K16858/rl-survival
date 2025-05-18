import os
import sys
import time
import numpy as np
import torch
from collections import deque
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from env.env import IslandEnvironment, TileType
from train.ppo_agent import PPOAgent
from train.utils import plot_training_progress, create_experiment_dir, save_hyperparameters
from train.train_ppo import preprocess_vision, preprocess_status

def visual_train_ppo(
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
    render_fps=10,
    experiment_name=None
):
    """
    Train PPO agent in island environment with real-time rendering
    """
    if experiment_name:
        exp_dir = os.path.join(model_dir, experiment_name)
    else:
        exp_dir = create_experiment_dir(model_dir)
    os.makedirs(exp_dir, exist_ok=True)
    print(f"Experiment directory: {exp_dir}")
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
    env = IslandEnvironment(size=env_size, seed=seed)
    vision_shape = (11, 11)
    obs = env.reset()
    vision = obs['vision']
    status = preprocess_status(obs, obs['position'])
    status_size = len(status)
    n_actions = len(env.action_space)
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
    best_score = float('-inf')
    scores = []
    scores_window = deque(maxlen=100)
    for epoch in range(1, n_epochs+1):
        obs = env.reset()
        vision = preprocess_vision(obs['vision'])
        status = preprocess_status(obs, obs['position'])
        score = 0
        for step in range(n_steps):
            action, log_prob, value = agent.act(vision, status)
            next_obs, reward, done, _ = env.step(action)
            next_vision = preprocess_vision(next_obs['vision'])
            next_status = preprocess_status(next_obs, next_obs['position'])
            agent.remember(vision, status, action, log_prob, value, reward, done)
            vision = next_vision
            status = next_status
            score += reward
            # --- ここで環境を表示 ---
            env.render()
            time.sleep(1.0 / render_fps)
            if step % 100 == 0:
                print(f"Epoch {epoch}/{n_epochs}, Step {step}/{n_steps}, Current score: {score:.2f}")
            if done:
                break
        agent.learn()
        scores_window.append(score)
        scores.append(score)
        if epoch % log_interval == 0:
            avg_score = np.mean(scores_window)
            print(f'Epoch {epoch}\tAverage Score: {avg_score:.2f}')
            plot_training_progress(
                scores, 
                window_size=min(100, len(scores)),
                save_path=os.path.join(exp_dir, f'scores_epoch_{epoch}.png')
            )
        if epoch % 10 == 0 or np.mean(scores_window) > best_score:
            if np.mean(scores_window) > best_score:
                best_score = np.mean(scores_window)
                agent.save(os.path.join(exp_dir, 'ppo_best.pth'))
                print(f"New best score: {best_score:.2f}, model saved.")
            if epoch % 10 == 0:
                agent.save(os.path.join(exp_dir, f'ppo_epoch_{epoch}.pth'))
    agent.save(os.path.join(exp_dir, 'ppo_final.pth'))
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

if __name__ == "__main__":
    visual_train_ppo(
        env_size=500,
        n_epochs=100,
        n_steps=2048,
        render_fps=10
    )
