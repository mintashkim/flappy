import os, sys
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(os.path.join(parent_dir))
sys.path.append(os.path.join(parent_dir, 'envs'))
# NOTE: Past Env
# from logs.PPO_22_BEST.flappy_env_joint_input import FlappyEnv
# NOTE: Current Env
from envs.flappy_env_joint_input import FlappyEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from envs.ppo.ppo import PPO # Customized
from stable_baselines3.common.evaluation import evaluate_policy
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
    save_path = os.path.join('saved_models/saved_model_'+experiment_id)
    loaded_model = PPO.load(save_path+"/best_model")
    
    # Environment parameters
    traj_type = args_dict['traj_type']
    render_mode = 'human' if args_dict['visualize'] else None
    env = FlappyEnv(render_mode=render_mode, trajectory_type=traj_type)
    env = VecMonitor(DummyVecEnv([lambda: env]))
    
    print("Evaluation start!")
    evaluate_policy(loaded_model, env, n_eval_episodes=10, render=True)
    env.close()

    # tensorboard --logdir logs/PPO_

main()