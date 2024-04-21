import os
import sys
sys.path.append('/home/mkim/flappy/flappy_v2')
from envs.ppo.ppo import PPO # Customized
# from stable_baselines3 import PPO # Naive
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.utils import set_random_seed
from envs.flappy_env_joint_input import FlappyEnv


log_path = os.path.join('logs')
save_path = os.path.join('saved_models/saved_models_PPO_server_2')

def create_env(rank: int, seed: int = 0):
    def _init():
        env = FlappyEnv()
        env.reset(seed=seed+rank)
        return env
    set_random_seed(seed)
    return _init

num_cpu = 4
env = VecMonitor(DummyVecEnv([create_env(i) for i in range(num_cpu)]))
# env = VecMonitor(DummyVecEnv([lambda: env]))

stop_callback = StopTrainingOnRewardThreshold(reward_threshold=20000, verbose=1)
eval_callback = EvalCallback(env,
                             callback_on_new_best=stop_callback,
                             eval_freq=10000,
                             best_model_save_path=save_path,
                             verbose=1)

# NOTE: if is_history: use 128 or more
# net_arch = {'pi': [128,128,128],
#             'vf': [128,128,128]}
# net_arch = {'pi': [256,128,128,64],
#             'vf': [256,128,128,64]}
net_arch = {'pi': [64,64,64],
            'vf': [64,64,64]}

def linear_schedule(initial_value):
    if isinstance(initial_value, str):
        initial_value = float(initial_value)
    def func(progress):
        return progress * initial_value
    return func

model = PPO('MlpPolicy', 
            env=env,
            learning_rate=3e-4,
            n_steps=2048, # The number of steps to run for each environment per update / 2048
            batch_size=256,
            gamma=0.99,  # 0.99 # look forward 1.65s
            gae_lambda=0.95,
            clip_range=linear_schedule(0.2),
            ent_coef=0.01, # Makes PPO explore
            verbose=1,
            policy_kwargs={'net_arch':net_arch},
            tensorboard_log=log_path,
            device='cuda')

model.learn(total_timesteps=1e+8, # The total number of samples (env steps) to train on
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