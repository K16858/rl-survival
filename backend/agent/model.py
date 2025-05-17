import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict

class ActorCritic(nn.Module):
    """
    アクターとクリティックを組み合わせたネットワーク
    """
    def __init__(self, 
                 vision_shape: Tuple[int, int],
                 status_size: int,
                 n_actions: int,
                 hidden_dim: int = 256):
        super(ActorCritic, self).__init__()
        
        # ビジョンフィールドを処理するCNN
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2)
        
        # CNNの出力サイズを計算
        h, w = vision_shape
        h_out = h // 4  # 2回のプーリングで1/4に
        w_out = w // 4
        cnn_output_dim = 64 * h_out * w_out
        
        # ステータスを処理する全結合層
        self.status_fc = nn.Linear(status_size, 64)
        
        # 特徴量を結合して処理する層
        self.combined_fc = nn.Linear(cnn_output_dim + 64, hidden_dim)
        
        # アクター（方策）
        self.actor = nn.Linear(hidden_dim, n_actions)
        
        # クリティック（価値関数）
        self.critic = nn.Linear(hidden_dim, 1)
        
    def forward_features(self, vision, status):
        """
        特徴抽出部分の順伝播
        """
        # ビジョン処理
        x = F.relu(self.conv1(vision))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        vision_features = x.view(x.size(0), -1)
        
        # ステータス処理
        status_features = F.relu(self.status_fc(status))
        
        # 特徴量の結合
        combined = torch.cat((vision_features, status_features), dim=1)
        features = F.relu(self.combined_fc(combined))
        
        return features
        
    def forward(self, vision, status):
        """
        順伝播計算
        """
        features = self.forward_features(vision, status)
        
        # アクター：行動の確率分布を出力
        action_probs = F.softmax(self.actor(features), dim=-1)
        
        # クリティック：状態価値を出力
        state_value = self.critic(features)
        
        return action_probs, state_value

