import os
import sys
sys.path.append('/Users/mintaekim/Desktop/HRL/Flappy/Integrated/Flappy_Integrated/flappy_v2')
from envs.ppo.ppo import PPO # Customized
# from stable_baselines3 import PPO # Naive
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
# NOTE: Past Env
from logs.PPO_22_TL.flappy_env_joint_input import FlappyEnv
# NOTE: Current Env
# from envs.flappy_env_joint_input import FlappyEnv


log_path = os.path.join('logs')
save_path = os.path.join('saved_models/saved_models_PPO_22_TL')
env = FlappyEnv(render_mode="human")
env = VecMonitor(DummyVecEnv([lambda: env]))

stop_callback = StopTrainingOnRewardThreshold(reward_threshold=10000, verbose=1)
eval_callback = EvalCallback(env,
                             callback_on_new_best=stop_callback,
                             eval_freq=10000,
                             best_model_save_path=save_path,
                             verbose=1)

# net_arch = {'pi': [512,256,256,128],
#             'vf': [512,256,256,128]}
# net_arch = {'pi': [64,128,128,64],
#             'vf': [64,128,128,64]}
# net_arch = {'pi': [128,128],
#             'vf': [128,128]}

loaded_model = PPO.load('saved_models/saved_models_PPO_22/best_model', env=env)

loaded_model.learn(total_timesteps=1e+7, # The total number of samples (env steps) to train on
                   progress_bar=True,
                   callback=eval_callback)

loaded_model.save(save_path+"/saved_models_PPO_22_TL_2")


####################################################
#################### Evaluation ####################
####################################################

obs_sample = loaded_model.env.observation_space.sample()

loaded_model = PPO.load(save_path+"/best_model")
print("Loaded model prediction: ")
print(loaded_model.predict(obs_sample, deterministic=True))

print("Evaluation start")
evaluate_policy(loaded_model, env, n_eval_episodes=5, render=True)
env.close()