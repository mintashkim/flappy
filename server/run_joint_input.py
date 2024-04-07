import os, sys
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(os.path.join(parent_dir))
sys.path.append(os.path.join(parent_dir, 'envs'))
from envs.ppo.ppo import PPO # Customized
# from stable_baselines3 import PPO # Naive
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.utils import set_random_seed
from envs.flappy_env_joint_input import FlappyEnv
from PIL import Image
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process some arguments')
    
    # Execution parameters
    parser.add_argument('--id', type=str, default='untitled', help='Provide experiment name and ID.')
    parser.add_argument('--num_steps', type=int, default=1e+9, help='Provide number of steps.')
    # parser.add_argument('--checkpoint', type=str, default=None, help='Loading pretrained model. Provide model path.')
    parser.add_argument('--device', type=str, default='mps', help='Provide device info.')
    parser.add_argument('--num_envs', type=int, default=8, help='Provide number of parallel environments.')
    # parser.add_argument('--test', type=str, default='False', help='Test. Default train.')
    # parser.add_argument('--record', type=str, default='False', help='Load Checkpoint.')
    # parser.add_argument('--task', type=str, default='JI', help='task type either JI or FDC') # JointInput or FDC
    
    # Environment parameters
    parser.add_argument('--traj_type', type=str, default='setpoint', help='Choose trajectory type.')
    parser.add_argument('--visualize', type=bool, default=False, help='Choose visualization option.')
    
    args = parser.parse_args()
    args_dict = vars(args)

    return args_dict

def main():
    args_dict = parse_arguments()
    print(args_dict)
    
    # Path
    experiment_id = args_dict['id']
    log_path = os.path.join('logs')
    save_path = os.path.join('saved_models/saved_model_'+experiment_id)
    
    # Environment parameters
    traj_type = args_dict['traj_type']
    render_mode = 'human' if args_dict['visualize'] else None
    
    # Parallel environment
    def create_env(seed=0):
        def _init():
            env = FlappyEnv(render_mode=render_mode, trajectory_type=traj_type, env_num=seed)
            return env
        set_random_seed(seed)
        return _init
    
    num_cpu = args_dict['num_envs']
    env = VecMonitor(DummyVecEnv([create_env(seed=i) for i in range(num_cpu)]))

    # Callbacks
    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=20000, verbose=1)
    eval_callback = EvalCallback(env,
                                 callback_on_new_best=stop_callback,
                                 eval_freq=100000,
                                 best_model_save_path=save_path,
                                 verbose=1)

    # Networks
    # NOTE: if is_history: use 128 or more
    # net_arch = {'pi': [128,128,128],
    #             'vf': [128,128,128]}
    # net_arch = {'pi': [256,128,128,64],
    #             'vf': [256,128,128,64]}
    net_arch = {'pi': [64,64,64],
                'vf': [64,64,64]}

    # PPO Modeling
    def linear_schedule(initial_value):
        if isinstance(initial_value, str):
            initial_value = float(initial_value)
        def func(progress):
            return progress * initial_value
        return func

    num_steps = args_dict['num_steps']
    device = args_dict['device']
    
    model = PPO('MlpPolicy', 
                env=env,
                learning_rate=3e-4,
                n_steps=2048*16, # The number of steps to run for each environment per update / 2048
                batch_size=256*16,
                gamma=0.99,  # 0.99 # look forward 1.65s
                gae_lambda=0.95,
                clip_range=linear_schedule(0.2),
                ent_coef=0.01, # Makes PPO explore
                verbose=1,
                policy_kwargs={'net_arch':net_arch},
                tensorboard_log=log_path,
                device=device)

    model.learn(total_timesteps=num_steps, # The total number of samples (env steps) to train on
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

main()