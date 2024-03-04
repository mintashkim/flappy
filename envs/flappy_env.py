# Helpers
import os
import sys
sys.path.append('/Users/mintaekim/Desktop/HRL/Flappy/Integrated/Flappy_Integrated/flappy_v3')
sys.path.append('/Users/mintaekim/Desktop/HRL/Flappy/Integrated/Flappy_Integrated/flappy_v3/envs')
import numpy as np
import jax.numpy as jnp
from jax import jit
from typing import Dict, Union
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
# Gym
import gymnasium as gym
from gymnasium.spaces import Box
from gymnasium import utils
from gymnasium.utils import seeding
import pygame
# Flappy
from dynamics import Flappy
from parameter import Simulation_Parameter
from aero_force import aero
from action_filter import ActionFilterButter
from env_randomize import EnvRandomizer
from utility_functions import *
import utility_trajectory as ut
from rotation_transformations import *
from R_body import R_body


DEFAULT_CAMERA_CONFIG = {"trackbodyid": 0, "distance": 12.0}
TRAJECTORY_TYPES = {"linear": 0, "circular": 1, "setpoint": 2}

class FlappyEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array", "depth_array"]}
    
    def __init__(
        self,
        max_timesteps = 10000,
        is_visual     = False,
        randomize     = False,
        debug         = False,
        lpf_action    = True,
        traj_type     = False,
        **kwargs
    ):
        # Dynamics simulator
        self.p = Simulation_Parameter()
        self.sim = Flappy(p=self.p, render=is_visual)

        # Frequency
        self.max_timesteps         = max_timesteps
        self.timestep              = 0
        self.sim_freq              = self.sim.freq # NOTE: 2000Hz for hard coding
        self.dt                    = self.sim.dt # NOTE: 1/2000s for hard coding
        self.policy_freq           = 30 # NOTE: 30Hz but the real control frequency might not be exactly 30Hz because we round up the num_sims_per_env_step
        self.num_sims_per_env_step = self.sim_freq // self.policy_freq # 2000//30 = 66
        self.secs_per_env_step     = self.num_sims_per_env_step * self.dt # 66/2000 = 0.033s
        self.policy_freq           = int(1.0/self.secs_per_env_step) # int(1000/33) = 30Hz

        self.is_visual          = is_visual
        self.randomize          = randomize
        self.debug              = debug
        self.is_plotting        = True
        self.traj_type          = traj_type
        self.noisy              = False
        self.randomize_dynamics = False # True to randomize dynamics
        self.lpf_action         = lpf_action # Low Pass Filter
        self.is_aero            = True

        # Observation, need to be reduce later for smoothness
        self.n_state            = 84 # NOTE: pos, vel, ang_vel, ori, ori_vel
        self.n_action           = 9  # NOTE: thrusts, f1, f2, f3, f4, f5, f6 # NOTE: u_gains
        self.history_len_short  = 4
        self.history_len_long   = 10
        self.history_len        = self.history_len_short
        self.previous_obs       = deque(maxlen=self.history_len)
        self.previous_act       = deque(maxlen=self.history_len)
        self.last_act_norm      = np.zeros(self.n_action)
        
        self.observation_space  = Box(low=-np.inf, high=np.inf, shape=(12,)) # NOTE: change to the actual number of obs to actor policy

        # NOTE: the low & high does not actually limit the actions output from MLP network, manually clip instead
        self.pos_lb = np.array([-5,-5,-5])
        self.pos_ub = np.array([5,5,5])
        self.displacement_bound = 5.0
        self.speed_bound = 100.0

        # Action Space
        # NOTE: Thrusts as actions
        # self.action_lower_bounds = np.array([-30,0,0,0,0,0,0])
        # self.action_upper_bounds = np.array([-30,1,1,1,1,0.5,0.5]) # Without FDC, ub of flapping freq is -30
        # self.action_bounds_scale = 0.0
        # self.action_lower_bounds_actual = np.concatenate([self.action_lower_bounds[0:5] + self.action_bounds_scale * self.action_upper_bounds[0:5], 
        #                                                   self.action_lower_bounds[5:7]])
        # self.action_upper_bounds_actual = np.concatenate([(1 - self.action_bounds_scale) * self.action_upper_bounds[0:5],
        #                                                   self.action_upper_bounds[5:7]])
        # NOTE: u_gain as actions
        self.action_lower_bounds_actual = np.array([8,  1,  0.1,2,  0.1,0.1,2,  2,  0.15])
        self.action_upper_bounds_actual = np.array([8.5,1.5,0.2,2.5,0.2,0.2,2.5,2.5,0.2])
        self.action_space = Box(low=self.action_lower_bounds_actual, high=self.action_upper_bounds_actual)

        # Info for normalizing the state
        self._init_action_filter()
        # self._init_env_randomizer() # NOTE: Take out dynamics randomization first 
        self._seed()
        self.reset()
        self._init_env()

    def _init_env(self):
        print("Environment created")
        action = self.action_space.sample()
        print("Sample action: {}".format(action))
        print("Action space: {}".format(self.action_space))
        print("Time step(dt): {}".format(self.dt))
        
    def _init_action_filter(self):
        self.action_filter = ActionFilterButter(
            lowcut        = None,
            highcut       = [4],
            sampling_rate = self.policy_freq,
            order         = 2,
            num_joints    = self.n_action,
        )

    def _seed(self, seed=None):
        self.np_random, _seeds = seeding.np_random(seed)
        return [seed]

    def reset(self, seed=None, randomize=None):
        if randomize is None: randomize = self.randomize
        self._reset_env(randomize)
        self.action_filter.reset()
        # self.env_randomizer.randomize_dynamics()
        # self._set_dynamics_properties()
        self._update_data(step=False)
        obs = self._get_obs()
        return obs, self.info
    
    def _reset_env(self, randomize=False):
        self.timestep    = 0
        self.time_in_sec = 0.0
        self.goal = np.concatenate([np.array([0.0, 0.0, 0.0]), np.zeros(3)]) # goal is x y z vx vy vz
        self.sim.reset()
        self.last_act   = np.zeros(self.n_action)
        self.reward     = None
        self.terminated = None
        self.info       = {}

    def _init_env_randomizer(self):
        self.env_randomizer = EnvRandomizer(self.sim)

    def _set_dynamics_properties(self):
        if self.randomize_dynamics:
            self.sim.set_dynamics()

    # TODO: Legacy
    def _get_obs(self):
        obs_curr = self.obs_states      
        obs_curr_gt = self.gt_states
        obs_command = self.goal
        return obs_curr

    def _act_norm2actual(self, act):
        return self.action_lower_bounds_actual + (act + 1)/2.0 * (self.action_upper_bounds_actual - self.action_lower_bounds_actual)

    def step(self, action, restore=False):
        assert action.shape[0] == self.n_action and -1.0 <= action.all() <= 1.0
        # action = self._act_norm2actual(action)
        if self.timestep == 0: self.action_filter.init_history(action)
        # post-process action
        if self.lpf_action: action_filtered = self.action_filter.filter(action)
        else: action_filtered = np.copy(action)
        for _ in range(self.num_sims_per_env_step):
            self.sim.step(action_filtered)

        obs = self._get_obs()
        reward, reward_dict = self._get_reward(action)
        self.info["reward_dict"] = reward_dict

        self._update_data(step=True)
        self.last_act_norm = action
        terminated = self._terminated(obs)
        truncated = False
        
        return obs, reward, terminated, truncated, self.info

    def _update_data(self, step=True):
        # NOTE: Need to be modified to obs states and ground truth states 
        self.obs_states = self.sim.get_obseverable()
        self.gt_states = self.sim.states
        if step:
            self.timestep += self.num_sims_per_env_step
            self.time_in_sec += self.secs_per_env_step
            # self.time_in_sec = self.sim.time
            # self.reference_generator.update_ref_env(self.time_in_sec)

    def _get_reward(self, action_normalized):
        names = ['position_error', 'velocity_error', 'angular_velocity', 'orientation_error', 'input', 'delta_acs']

        w_position             = 5.0
        w_velocity             = 1.0
        w_orientation          = 10.0
        w_orientation_velocity = 1.0
        w_input                = 5.0
        w_delta_act            = 0.1

        reward_weights = np.array([w_position, w_velocity, w_orientation, w_orientation_velocity, w_input, w_delta_act])
        weights = reward_weights / np.sum(reward_weights)  # weight can be adjusted later

        scale_pos       = 1.0
        scale_vel       = 1.0
        scale_ori       = 1.0
        scale_ori_vel   = 1.0
        scale_input     = 1.0 # Action already normalized
        scale_delta_act = 1.0

        desired_pos_norm     = np.array([0.0,0.0,0.0]).reshape(3,1)/5 # x y z 
        desired_vel_norm     = np.array([0.0,0.0,0.0]).reshape(3,1)/5 # vx vy vz
        desired_ori_norm     = np.array([0.0,0.0,0.0]).reshape(3,1)/np.pi # roll, pitch, yaw
        desired_ori_vel_norm = np.array([0.0,0.0,0.0]).reshape(3,1)/10 # roll_rate, pitch_rate, yaw_rate

        obs = self._get_obs()
        current_pos_norm     = obs[0:3]/5 # [-5,5] -> [-1,1]
        current_vel_norm     = obs[3:6]/5 # [-5,5] -> [-1,1]
        current_ori_norm     = obs[6:9]/np.pi
        current_ori_vel_norm = obs[9:12]/10 # [-10,10] -> [-1,1]

        pos_err       = np.linalg.norm(current_pos_norm - desired_pos_norm) # [0,1]
        vel_err       = np.linalg.norm(current_vel_norm - desired_vel_norm) # [0,1]
        ori_err       = np.linalg.norm(current_ori_norm - desired_ori_norm) # [0,1]
        ori_vel_err   = np.linalg.norm(current_ori_vel_norm - desired_ori_vel_norm) # [0,1]
        input_err     = np.linalg.norm(action_normalized) # It's not an error but let's just call it
        delta_act_err = np.linalg.norm(action_normalized - self.last_act_norm) # It's not an error but let's just call it

        rewards = np.exp(-np.array([scale_pos, scale_vel, scale_ori, scale_ori_vel, scale_input, scale_delta_act]
                         * np.array([pos_err, vel_err, ori_err, ori_vel_err, input_err, delta_act_err])))
        total_reward = np.sum(weights * rewards)
        reward_dict = dict(zip(names, weights * rewards))

        return total_reward, reward_dict

    def _terminated(self, obs):
        if not((obs[0:3] <= self.pos_ub).all() 
           and (obs[0:3] >= self.pos_lb).all()):
            print("Out of position bounds: {pos}  |  Timestep: {timestep}  |  Time: {time}s".format(pos=obs[0:3], timestep=self.timestep, time=round(self.time_in_sec,2)))
            return True
        if not(np.linalg.norm(obs[3:6]) <= self.speed_bound):
            print("Out of speed bounds: {vel}  |  Timestep: {timestep}  |  Time: {time}s".format(vel=obs[3:6], timestep=self.timestep, time=round(self.time_in_sec,2)))
            return True
        if self.timestep >= self.max_timesteps:
            print("Max step reached: Timestep: {timestep}  |  Time: {time}s".format(timestep=self.max_timesteps, time=round(self.time_in_sec,2)))
            return True
        else:
            return False

"""
if __name__ == "__main__":
    env = FlappyEnv()
    for _ in range(5):
        action = env.action_space.sample()
        tik = time.perf_counter()
        obs, reward, terminated, truncated, info = env.step(action)
        tok = time.perf_counter()
        print("Observation: {obs}".format(obs=obs))
        print("Reward: {reward}".format(reward=reward))
        print("Terminated: {terminated}".format(terminated=terminated))
        print("Step time: {}".format(tok-tik))
"""