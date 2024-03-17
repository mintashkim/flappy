import os
import sys
sys.path.append('/Users/mintaekim/Desktop/HRL/Flappy/Integrated/Flappy_Integrated/flappy_v2')
from envs.flappy_env_joint_input import FlappyEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from envs.ppo.ppo import PPO # Customized
from stable_baselines3.common.evaluation import evaluate_policy


env = FlappyEnv(render_mode="human")
env = VecMonitor(DummyVecEnv([lambda: env]))
save_path = os.path.join('saved_models/saved_models_PPO_9')
loaded_model = PPO.load(save_path+"/best_model")

print("Evaluation start!")
evaluate_policy(loaded_model, env, n_eval_episodes=10, render=True)
env.close()

# tensorboard --logdir logs/PPO_