import os
import sys
import time
import numpy as np
import torch
from collections import deque
import matplotlib.pyplot as plt
import argparse

# 親ディレクトリをパスに追加してモジュールをインポート
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from env.env import IslandEnvironment, TileType
from agent.ppo_agent import PPOAgent
from agent.utils import plot_training_progress, create_experiment_dir, save_hyperparameters

def preprocess_vision(vision):
    """ビジョン配列をCNN入力用に正規化"""
    # 0-1に正規化
    return vision.astype(np.float32) / max(1, vision.max())

def preprocess_status(obs, agent_pos):
    """状態特徴量の抽出と正規化"""
    # 現在位置のタイルタイプを取得
    current_tile = obs['current_tile']
    
    # タイルタイプをOne-hotエンコーディング
    tile_type_onehot = np.zeros(7)  # 7つのタイルタイプを想定
    if current_tile < 7:
        tile_type_onehot[current_tile] = 1.0
    
    # 位置の正規化
    norm_pos = np.array(agent_pos) / 500.0  # マップサイズ最大500を想定
    
    # 特徴量の結合
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
    PPOエージェントを島環境で訓練する
    
    Args:
        env_size: 島マップのサイズ
        n_epochs: 訓練エポック数
        n_steps: 各エポックのステップ数
        model_dir: モデルを保存するディレクトリ
        log_interval: 進捗をログする間隔
        gamma: 割引率
        gae_lambda: GAE λパラメータ
        policy_clip: 方策クリッピングパラメータ
        batch_size: バッチサイズ
        n_updates: 各バッチで更新を繰り返す回数
        lr: 学習率
        seed: 乱数シード
        render: 環境を描画するかどうか
        experiment_name: 実験の名前（Noneの場合はタイムスタンプ）
        
    Returns:
        scores: 各エポックのスコアリスト
    """
    # 実験ディレクトリの作成
    if experiment_name:
        exp_dir = os.path.join(model_dir, experiment_name)
    else:
        exp_dir = create_experiment_dir(model_dir)
    
    os.makedirs(exp_dir, exist_ok=True)
    print(f"実験ディレクトリ: {exp_dir}")
    
    # ハイパーパラメータの保存
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
    
    # 環境の作成
    env = IslandEnvironment(size=env_size, seed=seed)
    
    # ビジョンフィールドの形状
    vision_shape = (11, 11)  # IslandEnvironmentの_get_observationから
    
    # ステータスサイズの取得
    obs = env.reset()
    vision = obs['vision']
    status = preprocess_status(obs, obs['position'])
    status_size = len(status)
    
    # アクション数
    n_actions = len(env.action_space)
    
    # エージェントの作成
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
    
    # スコア記録
    best_score = float('-inf')
    scores = []
    scores_window = deque(maxlen=100)  # 直近100エポックのスコア
    
    # 訓練ループ
    for epoch in range(1, n_epochs+1):
        obs = env.reset()
        vision = preprocess_vision(obs['vision'])
        status = preprocess_status(obs, obs['position'])
        score = 0
        
        # 1エポック分のステップを実行
        for step in range(n_steps):
            # 行動選択
            action, log_prob, value = agent.act(vision, status)
            
            # 行動実行
            next_obs, reward, done, _ = env.step(action)
            
            # 前処理
            next_vision = preprocess_vision(next_obs['vision'])
            next_status = preprocess_status(next_obs, next_obs['position'])
            
            # データをメモリに保存
            agent.remember(vision, status, action, log_prob, value, reward, done)
            
            # 状態の更新
            vision = next_vision
            status = next_status
            score += reward
            
            # レンダリング
            if render:
                env.render()
                time.sleep(0.01)
            
            if step % 100 == 0:
                print(f"Epoch {epoch}/{n_epochs}, Step {step}/{n_steps}, Current score: {score:.2f}")
            
            if done:
                break
        
        # エポック終了後に学習実行
        agent.learn()
        
        # スコア記録
        scores_window.append(score)
        scores.append(score)
        
        # 進捗表示
        if epoch % log_interval == 0:
            avg_score = np.mean(scores_window)
            print(f'Epoch {epoch}\tAverage Score: {avg_score:.2f}')
            
            # スコア推移プロット
            plot_training_progress(
                scores, 
                window_size=min(100, len(scores)),
                save_path=os.path.join(exp_dir, f'scores_epoch_{epoch}.png')
            )
        
        # ベストモデルの保存
        if epoch % 10 == 0 or np.mean(scores_window) > best_score:
            if np.mean(scores_window) > best_score:
                best_score = np.mean(scores_window)
                agent.save(os.path.join(exp_dir, 'ppo_best.pth'))
                print(f"新しいベストスコア: {best_score:.2f}, モデルを保存しました。")
            
            # 定期保存
            if epoch % 10 == 0:
                agent.save(os.path.join(exp_dir, f'ppo_epoch_{epoch}.pth'))
    
    # 最終モデルの保存
    agent.save(os.path.join(exp_dir, 'ppo_final.pth'))
    
    # 最終スコアプロットの保存
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(scores)), scores)
    plt.plot(np.arange(len(scores)), 
             [np.mean(scores[max(0, i-100):i+1]) for i in range(len(scores))],
             label='100エポック移動平均')
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
    訓練済みPPOエージェントのテスト
    
    Args:
        model_path: テストするモデルのパス
        env_size: 島マップのサイズ
        n_episodes: テストエピソード数
        max_steps: 1エピソードの最大ステップ数
        render: 環境を描画するかどうか
        fps: 描画のフレームレート
    """
    # 環境の作成
    env = IslandEnvironment(size=env_size, seed=42)
    
    # ビジョンフィールドの形状
    vision_shape = (11, 11)
    
    # ステータスサイズの取得
    obs = env.reset()
    vision = obs['vision']
    status = preprocess_status(obs, obs['position'])
    status_size = len(status)
    
    # アクション数
    n_actions = len(env.action_space)
    
    # エージェントの作成
    agent = PPOAgent(
        vision_shape=vision_shape,
        status_size=status_size,
        n_actions=n_actions
    )
    
    # モデルの読み込み
    agent.load(model_path)
    
    # スコア記録
    scores = []
    
    # テストループ
    for i_episode in range(1, n_episodes+1):
        obs = env.reset()
        vision = preprocess_vision(obs['vision'])
        status = preprocess_status(obs, obs['position'])
        score = 0
        
        for t in range(max_steps):
            # レンダリング
            if render:
                env.render()
                time.sleep(1.0 / fps)
            
            # 行動選択
            action, _, _ = agent.act(vision, status)
            
            # 行動実行
            next_obs, reward, done, _ = env.step(action)
            
            # 前処理
            next_vision = preprocess_vision(next_obs['vision'])
            next_status = preprocess_status(next_obs, next_obs['position'])
            
            # 状態の更新
            vision = next_vision
            status = next_status
            score += reward
            
            if done:
                break
        
        scores.append(score)
        print(f'エピソード {i_episode}\tスコア: {score:.2f}')
    
    # 環境終了
    env.close()
    
    # 平均スコアの表示
    avg_score = np.mean(scores)
    print(f'平均スコア: {avg_score:.2f}')
    return avg_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='島環境でPPOエージェントの訓練またはテストを行う')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                        help='モード: trainまたはtest')
    parser.add_argument('--env_size', type=int, default=200,
                        help='島マップのサイズ')
    parser.add_argument('--epochs', type=int, default=100,
                        help='訓練エポック数')
    parser.add_argument('--steps', type=int, default=2048,
                        help='1エポックのステップ数')
    parser.add_argument('--model_dir', type=str, default='models_ppo',
                        help='モデルを保存/読み込むディレクトリ')
    parser.add_argument('--model_path', type=str, default='ppo_final.pth',
                        help='テスト時のモデルファイル名')
    parser.add_argument('--render', action='store_true',
                        help='環境を描画する')
    parser.add_argument('--experiment', type=str, default=None,
                        help='実験名（ディレクトリ名）')
    
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
