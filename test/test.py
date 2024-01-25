import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from stable_baselines3 import PPO
from flappy_env import FlappyEnv

PPO_path = PPO_path = os.path.join('..', 'train', 'saved_models', 'best_model')
env = FlappyEnv(render_mode="human")

model = PPO.load(PPO_path, env=env)
model.learn(total_timesteps=100000)