import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict

class ActorCritic(nn.Module):
    """
    Combined network of actor and critic
    """
    def __init__(self, 
                 vision_shape: Tuple[int, int],
                 status_size: int,
                 n_actions: int,
                 hidden_dim: int = 256):
        super(ActorCritic, self).__init__()
        
        # CNN for processing vision field
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2)
        
        # Calculate CNN output size
        h, w = vision_shape
        h_out = h // 4  # 1/4 after two pooling operations
        w_out = w // 4
        cnn_output_dim = 64 * h_out * w_out
        
        # Fully connected layer to process status
        self.status_fc = nn.Linear(status_size, 64)
        
        # Layer to process combined features
        self.combined_fc = nn.Linear(cnn_output_dim + 64, hidden_dim)
        
        # Actor (policy)
        self.actor = nn.Linear(hidden_dim, n_actions)
        
        # Critic (value function)
        self.critic = nn.Linear(hidden_dim, 1)
        
    def forward_features(self, vision, status):
        """
        Forward propagation for feature extraction part
        """
        # Vision processing
        x = F.relu(self.conv1(vision))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        vision_features = x.view(x.size(0), -1)
        
        # Status processing
        status_features = F.relu(self.status_fc(status))
        
        # Combining features
        combined = torch.cat((vision_features, status_features), dim=1)
        features = F.relu(self.combined_fc(combined))
        
        return features
        
    def forward(self, vision, status):
        """
        Forward propagation calculation
        """
        features = self.forward_features(vision, status)
        
        # Actor: output probability distribution of actions
        action_probs = F.softmax(self.actor(features), dim=-1)
        
        # Critic: output state value
        state_value = self.critic(features)
        
        return action_probs, state_value

