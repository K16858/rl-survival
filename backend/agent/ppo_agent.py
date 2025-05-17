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
        PPOエージェントの初期化
        
        Args:
            vision_shape: ビジョンフィールドの形状 (height, width)
            status_size: ステータス特徴量の数
            n_actions: 可能なアクションの数
            gamma: 割引率
            gae_lambda: GAE（Generalized Advantage Estimation）のλパラメータ
            policy_clip: 方策更新のクリッピングパラメータ
            batch_size: バッチサイズ
            n_epochs: 各更新で学習を繰り返すエポック数
            lr: 学習率
            seed: ランダムシード
            device: 学習に使用するデバイス ('cpu', 'cuda', or 'auto')
        """
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.vision_shape = vision_shape
        self.status_size = status_size
        self.n_actions = n_actions
        
        # 乱数シードの設定
        random.seed(seed)
        torch.manual_seed(seed)
        
        # デバイスの決定
        if device == 'auto':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # アクタークリティックネットワークの作成
        self.actor_critic = ActorCritic(
            vision_shape=vision_shape,
            status_size=status_size,
            n_actions=n_actions
        ).to(self.device)
        
        # オプティマイザの設定
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        
        # メモリの初期化
        self.memory = PPOMemory(batch_size)
        
    def act(self, vision, status):
        """
        与えられた状態から行動を選択
        
        Args:
            vision: ビジョンフィールド
            status: ステータス
            
        Returns:
            選択されたアクション、行動確率、推定された状態価値
        """
        # バッチ次元の追加
        if len(vision.shape) == 2:  # [height, width]
            vision = np.expand_dims(vision, axis=0)  # [1, height, width]
        if len(vision.shape) == 3:  # [1, height, width]
            vision = np.expand_dims(vision, axis=0)  # [1, 1, height, width]
        if len(status.shape) == 1:  # [status_size]
            status = np.expand_dims(status, axis=0)  # [1, status_size]
        
        # テンソルに変換
        vision = torch.from_numpy(vision).float().to(self.device)
        status = torch.from_numpy(status).float().to(self.device)
        
        # ネットワーク推論
        self.actor_critic.eval()
        with torch.no_grad():
            action_probs, state_value = self.actor_critic(vision, status)
        self.actor_critic.train()
        
        # 確率分布の作成
        dist = Categorical(action_probs)
        
        # アクションのサンプリング
        action = dist.sample()
        
        # 選択したアクションの確率の対数を保存
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item(), state_value.item()
    
    def remember(self, vision, status, action, prob, val, reward, done):
        """メモリにトランジションを保存"""
        self.memory.store(vision, status, action, prob, val, reward, done)
    
    def learn(self):
        """保存したトランジションを使って学習を行う"""
        # エピソードがない場合は学習しない
        if len(self.memory.visions) == 0:
            return
            
        # すべてのエピソードデータをテンソルに変換
        device = self.device
        visions = torch.from_numpy(np.array(self.memory.visions)).float().to(device)
        statuses = torch.from_numpy(np.array(self.memory.statuses)).float().to(device)
        actions = torch.tensor(self.memory.actions, dtype=torch.long).to(device)
        old_probs = torch.tensor(self.memory.probs, dtype=torch.float).to(device)
        vals = torch.tensor(self.memory.vals, dtype=torch.float).to(device)
        
        # アドバンテージの計算
        rewards = np.array(self.memory.rewards)
        dones = np.array(self.memory.dones)
        
        advantages = np.zeros(len(rewards), dtype=np.float32)
        
        # GAEの計算
        for t in range(len(rewards)-1):
            discount = 1
            a_t = 0
            for k in range(t, len(rewards)-1):
                a_t += discount * (rewards[k] + self.gamma * vals[k+1] * (1 - dones[k]) - vals[k])
                discount *= self.gamma * self.gae_lambda * (1 - dones[k])
            advantages[t] = a_t
        
        advantages = torch.tensor(advantages, dtype=torch.float).to(device)
        
        # PPO更新ループ
        for _ in range(self.n_epochs):
            # バッチを生成
            batches = self.memory.generate_batches()
            
            for batch in batches:
                # 現在の方策と価値を計算
                action_probs, state_values = self.actor_critic(visions[batch], statuses[batch])
                state_values = state_values.squeeze()
                
                # 確率分布を作成
                dist = Categorical(action_probs)
                
                # 新しい行動確率の計算
                new_probs = dist.log_prob(actions[batch])
                
                # 確率比率を計算（新しい確率÷古い確率）
                prob_ratio = torch.exp(new_probs - old_probs[batch])
                
                # PPOの目的関数（クリッピングあり）
                weighted_probs = advantages[batch] * prob_ratio
                clipped_probs = advantages[batch] * torch.clamp(prob_ratio, 1-self.policy_clip, 1+self.policy_clip)
                actor_loss = -torch.min(weighted_probs, clipped_probs).mean()
                
                # クリティックの損失関数（MSE）
                returns = advantages[batch] + vals[batch]
                critic_loss = F.mse_loss(state_values, returns)
                
                # 合計損失
                total_loss = actor_loss + 0.5 * critic_loss
                
                # 勾配の更新
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
                self.optimizer.step()
                
        # 学習後にメモリをクリア
        self.memory.clear()

    def save(self, path: str):
        """モデルの保存"""
        torch.save({
            'model_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
    
    def load(self, path: str):
        """モデルの読み込み"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
