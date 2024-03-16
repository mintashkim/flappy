import os
import sys
sys.path.append('/Users/mintaekim/Desktop/HRL/Flappy/Integrated/Flappy_Integrated/flappy_v2')
from envs.ppo.ppo import PPO # Customized
# from stable_baselines3 import PPO # Naive
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from envs.flappy_env_joint_input import FlappyEnv


log_path = os.path.join('logs')
save_path = os.path.join('saved_models/saved_models_joint_input_5')
env = FlappyEnv(render_mode="human")
env = VecMonitor(DummyVecEnv([lambda: env]))

stop_callback = StopTrainingOnRewardThreshold(reward_threshold=3000, verbose=1)
eval_callback = EvalCallback(env,
                             callback_on_new_best=stop_callback,
                             eval_freq=10000,
                             best_model_save_path=save_path,
                             verbose=1)

# net_arch = {'pi': [512,256,256,128],
#             'vf': [512,256,256,128]}
# net_arch = {'pi': [64,128,128,64],
#             'vf': [64,128,128,64]}
net_arch = {'pi': [128,256,256,128],
            'vf': [128,256,256,128]}

model = PPO('MlpPolicy', 
            env=env,
            learning_rate=1e-4,
            n_steps=256, # The number of steps to run for each environment per update / 2048
            batch_size=256,
            gamma=0.99,  # 0.99 # look forward 1.65s
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01, # Makes PPO explore
            verbose=1,
            policy_kwargs={'net_arch':net_arch},
            tensorboard_log=log_path,
            device='mps'
            )

model.learn(total_timesteps=1e+7, # The total number of samples (env steps) to train on
            progress_bar=True,
            callback=eval_callback)

model.save(save_path)


####################################################
#################### Evaluation ####################
####################################################

obs_sample = model.env.observation_space.sample()

print("Pre saved model prediction: ")
print(model.predict(obs_sample, deterministic=True))
del model # delete trained model to demonstrate loading

loaded_model = PPO.load(save_path+"/best_model")
print("Loaded model prediction: ")
print(loaded_model.predict(obs_sample, deterministic=True))

print("Evaluation start")
evaluate_policy(loaded_model, env, n_eval_episodes=5, render=True)
env.close()