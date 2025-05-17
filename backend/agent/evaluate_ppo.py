import os
import sys
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
import pygame

# 親ディレクトリをパスに追加
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
    訓練済みPPOエージェントの評価と結果の保存
    
    Args:
        model_path: モデルファイルのパス
        env_size: 環境のサイズ
        n_episodes: 評価するエピソード数
        max_steps: 1エピソードあたりの最大ステップ数
        render: 環境を描画するかどうか
        fps: 描画のフレームレート
        output_dir: 評価結果を保存するディレクトリ
    """
    # 出力ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)
    
    # 環境の作成
    env = IslandEnvironment(size=env_size, seed=42, tile_size=8)
    
    # ビジョンフィールドの形状
    vision_shape = (11, 11)
    
    # ステータスサイズの取得
    obs = env.reset()
    vision = obs['vision']
    status = preprocess_status(obs, obs['position'])
    status_size = len(status)
    
    # アクション数
    n_actions = len(env.action_space)
    
    # デバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # エージェントの作成
    agent = PPOAgent(
        vision_shape=vision_shape,
        status_size=status_size,
        n_actions=n_actions,
        device=device
    )
    
    # モデルの読み込み
    agent.load(model_path)
    
    # 評価指標の記録
    episode_scores = []
    episode_steps = []
    episode_tiles_visited = []
    episode_river_visits = []
    
    # 複数エピソードで評価
    for i_episode in range(1, n_episodes + 1):
        print(f"評価エピソード {i_episode}/{n_episodes} を開始")
        
        # 環境のリセット
        obs = env.reset()
        vision = preprocess_vision(obs['vision'])
        status = preprocess_status(obs, obs['position'])
        score = 0
        
        # このエピソードで訪問したタイルを記録
        tiles_visited = set()
        river_visits = 0
        
        # 軌跡の記録
        trajectory = [env.agent_pos]
        
        # エピソード開始
        for t in range(1, max_steps + 1):
            # 行動選択
            action, _, _ = agent.act(vision, status)
            
            # 行動実行
            next_obs, reward, done, _ = env.step(action)
            
            # 軌跡に現在位置を追加
            trajectory.append(env.agent_pos)
            
            # 評価指標の記録
            score += reward
            i, j = map(int, env.agent_pos)
            tiles_visited.add((i, j))
            if env.map[i, j] == TileType.RIVER:
                river_visits += 1
            
            # 次の状態の前処理
            next_vision = preprocess_vision(next_obs['vision'])
            next_status = preprocess_status(next_obs, next_obs['position'])
            
            # 状態の更新
            vision = next_vision
            status = next_status
            
            # 描画
            if render:
                env.render()
                time.sleep(1.0 / fps)
            
            # 100ステップごとにスクリーンショットを保存
            if t % 100 == 0 and render:
                # スクリーンショットの保存（pygameが初期化されている場合）
                if env.is_initialized:
                    pygame_img_path = os.path.join(output_dir, f"episode_{i_episode}_step_{t}.png")
                    pygame.image.save(env.screen, pygame_img_path)
            
            # 進捗表示
            if t % 100 == 0:
                print(f"ステップ {t}/{max_steps}, スコア: {score:.2f}")
            
            if done:
                break
        
        # エピソード結果の記録
        episode_scores.append(score)
        episode_steps.append(t)
        episode_tiles_visited.append(len(tiles_visited))
        episode_river_visits.append(river_visits)
        
        # エピソードのサマリー表示
        print(f"エピソード {i_episode} を {t} ステップで完了")
        print(f"スコア: {score:.2f}")
        print(f"ユニークな訪問タイル数: {len(tiles_visited)}")
        print(f"川の訪問回数: {river_visits}")
        print("------------------------------")
        
        # エージェントの最終ビジョンを保存
        vision_path = os.path.join(output_dir, f"episode_{i_episode}_final_vision.png")
        plot_vision(obs['vision'], save_path=vision_path)
        
        # エピソードの軌跡を描画
        plt.figure(figsize=(10, 10))
        # まずマップの背景を描画
        map_img = np.zeros((env_size, env_size, 3), dtype=np.uint8)
        for i in range(env_size):
            for j in range(env_size):
                color = TileType.get_color(env.map[i, j])
                map_img[i, j] = color
        
        plt.imshow(map_img)
        
        # 軌跡を描画
        traj_x = [pos[1] for pos in trajectory]
        traj_y = [pos[0] for pos in trajectory]
        plt.plot(traj_x, traj_y, 'r-', linewidth=2, alpha=0.7)
        plt.plot(traj_x[0], traj_y[0], 'go', markersize=8)  # 開始位置
        plt.plot(traj_x[-1], traj_y[-1], 'ro', markersize=8)  # 終了位置
        
        plt.title(f"エピソード {i_episode} の軌跡")
        plt.savefig(os.path.join(output_dir, f"episode_{i_episode}_trajectory.png"))
        plt.close()
    
    # 環境を閉じる
    env.close()
    
    # 統計情報の計算
    avg_score = np.mean(episode_scores)
    avg_steps = np.mean(episode_steps)
    avg_tiles = np.mean(episode_tiles_visited)
    avg_river_visits = np.mean(episode_river_visits)
    
    # サマリーの表示
    print("===== 評価サマリー =====")
    print(f"平均スコア: {avg_score:.2f}")
    print(f"平均ステップ数: {avg_steps:.1f}")
    print(f"平均ユニーク訪問タイル数: {avg_tiles:.1f}")
    print(f"平均川訪問回数: {avg_river_visits:.2f}")
    
    # サマリーをファイルに保存
    with open(os.path.join(output_dir, "evaluation_summary.txt"), "w") as f:
        f.write("===== 評価サマリー =====\n")
        f.write(f"モデル: {model_path}\n")
        f.write(f"環境サイズ: {env_size}\n")
        f.write(f"エピソード数: {n_episodes}\n")
        f.write(f"最大ステップ数: {max_steps}\n\n")
        f.write(f"平均スコア: {avg_score:.2f}\n")
        f.write(f"平均ステップ数: {avg_steps:.1f}\n")
        f.write(f"平均ユニーク訪問タイル数: {avg_tiles:.1f}\n")
        f.write(f"平均川訪問回数: {avg_river_visits:.2f}\n\n")
        f.write("エピソード詳細:\n")
        for i in range(n_episodes):
            f.write(f"エピソード {i+1}: スコア={episode_scores[i]:.2f}, "
                    f"ステップ数={episode_steps[i]}, "
                    f"訪問タイル数={episode_tiles_visited[i]}, "
                    f"川訪問回数={episode_river_visits[i]}\n")
    
    # スコア分布の描画
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, n_episodes+1), episode_scores)
    plt.xlabel('エピソード')
    plt.ylabel('スコア')
    plt.title('エピソードごとの評価スコア')
    plt.savefig(os.path.join(output_dir, "score_distribution.png"))
    
    return avg_score, avg_steps, avg_tiles, avg_river_visits

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='訓練済みPPOエージェントの評価')
    parser.add_argument('--model', type=str, required=True, 
                        help='モデルファイルのパス')
    parser.add_argument('--env_size', type=int, default=200, 
                        help='環境のサイズ')
    parser.add_argument('--episodes', type=int, default=5, 
                        help='評価するエピソード数')
    parser.add_argument('--max_steps', type=int, default=500, 
                        help='1エピソードあたりの最大ステップ数')
    parser.add_argument('--no_render', action='store_true', 
                        help='描画を無効にする')
    parser.add_argument('--fps', type=int, default=30, 
                        help='描画のフレームレート')
    parser.add_argument('--output', type=str, default='ppo_evaluation_results', 
                        help='評価結果を保存するディレクトリ')
    
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
