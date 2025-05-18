import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from typing import Dict, List, Tuple, Any
import random

from .model import ActorCritic
from .ppo_memory import PPOMemory

class PPOAgent:
    """PPO (Proximal Policy Optimization) Agent"""
    
    def __init__(self,
                 vision_shape: Tuple[int, int],
                 status_size: int,
                 n_actions: int,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 policy_clip: float = 0.2,
                 batch_size: int = 64,
                 n_epochs: int = 10,
                 lr: float = 3e-4,
                 seed: int = 42,
                 device: str = 'auto'):
        """
        Initialize PPO agent
        
        Args:
            vision_shape: Shape of the vision field (height, width)
            status_size: Number of status features
            n_actions: Number of possible actions
            gamma: Discount factor
            gae_lambda: Lambda parameter for GAE (Generalized Advantage Estimation)
            policy_clip: Clipping parameter for policy update
            batch_size: Batch size
            n_epochs: Number of epochs to repeat learning on each update
            lr: Learning rate
            seed: Random seed
            device: Device used for learning ('cpu', 'cuda', or 'auto')
        """
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.vision_shape = vision_shape
        self.status_size = status_size
        self.n_actions = n_actions
        
        # Setting random seed
        random.seed(seed)
        torch.manual_seed(seed)
        
        # Determine device
        if device == 'auto':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Create actor-critic network
        self.actor_critic = ActorCritic(
            vision_shape=vision_shape,
            status_size=status_size,
            n_actions=n_actions
        ).to(self.device)
        
        # Setup optimizer
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        
        # Initialize memory
        self.memory = PPOMemory(batch_size)
        
    def act(self, vision, status):
        """
        Select an action based on the given state
        
        Args:
            vision: Vision field
            status: Status
            
        Returns:
            Selected action, action probability, and estimated state value
        """
        # Add batch dimension
        if len(vision.shape) == 2:  # [height, width]
            vision = np.expand_dims(vision, axis=0)  # [1, height, width]
        if len(vision.shape) == 3:  # [1, height, width]
            vision = np.expand_dims(vision, axis=0)  # [1, 1, height, width]
        if len(status.shape) == 1:  # [status_size]
            status = np.expand_dims(status, axis=0)  # [1, status_size]
        
        # Convert to tensor
        vision = torch.from_numpy(vision).float().to(self.device)
        status = torch.from_numpy(status).float().to(self.device)
        
        # Network inference
        self.actor_critic.eval()
        with torch.no_grad():
            action_probs, state_value = self.actor_critic(vision, status)
        self.actor_critic.train()
        
        # Create probability distribution
        dist = Categorical(action_probs)
        
        # Sample action
        action = dist.sample()
        
        # Save log probability of selected action
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item(), state_value.item()
    
    def remember(self, vision, status, action, prob, val, reward, done):
        """Store transition in memory"""
        self.memory.store(vision, status, action, prob, val, reward, done)
    
    def learn(self):
        """Learn from stored transitions"""
        # Skip learning if no episodes
        if len(self.memory.visions) == 0:
            return
            
        # Convert all episode data to tensors
        device = self.device
        visions = torch.from_numpy(np.array(self.memory.visions)).float().to(device)
        statuses = torch.from_numpy(np.array(self.memory.statuses)).float().to(device)
        actions = torch.tensor(self.memory.actions, dtype=torch.long).to(device)
        old_probs = torch.tensor(self.memory.probs, dtype=torch.float).to(device)
        vals = torch.tensor(self.memory.vals, dtype=torch.float).to(device)
        
        # Calculate advantage
        rewards = np.array(self.memory.rewards)
        dones = np.array(self.memory.dones)
        
        advantages = np.zeros(len(rewards), dtype=np.float32)
        
        # Calculate GAE
        for t in range(len(rewards)-1):
            discount = 1
            a_t = 0
            for k in range(t, len(rewards)-1):
                a_t += discount * (rewards[k] + self.gamma * vals[k+1] * (1 - dones[k]) - vals[k])
                discount *= self.gamma * self.gae_lambda * (1 - dones[k])
            advantages[t] = a_t
        
        advantages = torch.tensor(advantages, dtype=torch.float).to(device)
        
        # PPO update loop
        for _ in range(self.n_epochs):
            # Generate batches
            batches = self.memory.generate_batches()
            
            for batch in batches:
                # Calculate current policy and value
                action_probs, state_values = self.actor_critic(visions[batch], statuses[batch])
                state_values = state_values.squeeze()
                
                # Create probability distribution
                dist = Categorical(action_probs)
                
                # Calculate new action probabilities
                new_probs = dist.log_prob(actions[batch])
                
                # Calculate probability ratio (new probability ÷ old probability)
                prob_ratio = torch.exp(new_probs - old_probs[batch])
                
                # PPO objective function (with clipping)
                weighted_probs = advantages[batch] * prob_ratio
                clipped_probs = advantages[batch] * torch.clamp(prob_ratio, 1-self.policy_clip, 1+self.policy_clip)
                actor_loss = -torch.min(weighted_probs, clipped_probs).mean()
                
                # Critic loss function (MSE)
                returns = advantages[batch] + vals[batch]
                critic_loss = F.mse_loss(state_values, returns)
                
                # Total loss
                total_loss = actor_loss + 0.5 * critic_loss
                
                # Update gradients
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
                self.optimizer.step()
                
        # Clear memory after learning
        self.memory.clear()

    def save(self, path: str):
        """Save the model"""
        torch.save({
            'model_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
    
    def load(self, path: str):
        """Load the model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        