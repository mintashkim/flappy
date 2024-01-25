import os
import numpy as np
from typing import Dict, Union
from collections import deque

from gymnasium import utils
from gymnasium.spaces import Box
from gymnasium.utils import seeding

from mujoco_env import MujocoEnv
from parameter import Simulation_Parameter
from dynamics import Flappy
from rotation_transformations import *


DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 0,
    "distance": 5.0,
}

class FlappyEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(
        self,
        xml_file: str = "Flappy_v3.xml",
        frame_skip: int = 2,
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        reset_noise_scale: float = 0.01,
        **kwargs,
    ):
        utils.EzPickle.__init__(self, xml_file, frame_skip, reset_noise_scale, **kwargs)
        observation_space = Box(low=-np.inf, high=np.inf, shape=(41,), dtype=np.float64)

        self._reset_noise_scale = reset_noise_scale

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=observation_space,
            default_camera_config=default_camera_config,
            **kwargs,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        self.observation_structure = {
            "qpos": self.data.qpos.size,
            "qvel": self.data.qvel.size,
        }

        p = Simulation_Parameter()
        self.sim = Flappy(p=p)
        
        # Frequency
        self.sim_freq              = self.sim.freq # NOTE: set to sim.step dt 
        self.policy_freq           = 30 # 30Hz but the real control frequency is not exactly 30Hz because we round up the num_sims_per_env_step
        self.num_sims_per_env_step = self.sim_freq // self.policy_freq
        self.secs_per_env_step     = self.num_sims_per_env_step / self.sim_freq
        self.policy_freq           = int(1.0/self.secs_per_env_step)

        # History
        self.history_len_short  = 4
        self.history_len_long   = 10
        self.history_len        = self.history_len_short
        self.previous_obs = deque(maxlen=self.history_len)
        self.previous_acs = deque(maxlen=self.history_len)
        self.last_acs = np.zeros(self.action_space.shape)

        # self.traj_type        = traj_type
        self.randomize_dynamics = False # True to randomize dynamics
        # self.lpf_action       = lpf_action # action filter


        # State bounds
        self.pos_lb = np.array([-4, -4, 0.3])  # fight space dimensions: xyz
        self.pos_ub = np.array([4, 4, 4])  # fight space dimensions
        self.vel_lb = np.array([-2, -2, -2])
        self.vel_ub = np.array([2, 2, 2])

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self.data.ctrl[0] = 1
        self.do_simulation(action, self.frame_skip)

        observation = self._get_obs()
        terminated = bool(not np.isfinite(observation).all()
                          or (self.data.qpos[0] < self.pos_lb[0]) or (self.data.qpos[0] > self.pos_ub[0])
                          or (self.data.qpos[1] < self.pos_lb[1]) or (self.data.qpos[1] > self.pos_ub[1])
                          or (self.data.qpos[2] < self.pos_lb[2]) or (self.data.qpos[2] > self.pos_ub[2])
                          # or (self.data.qvel[0] < self.vel_lb[0]) or (self.data.qvel[0] > self.vel_ub[0])
                          # or (self.data.qvel[1] < self.vel_lb[1]) or (self.data.qvel[1] > self.vel_ub[1])
                          # or (self.data.qvel[2] < self.vel_lb[2]) or (self.data.qvel[2] > self.vel_ub[2])
                          )

        reward, reward_dict = self._get_reward(action)
        info = {"reward_survive": reward}
        # print(reward)

        if self.render_mode == "human":
            self.render()

        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        self.last_acs = action
        return observation, reward, terminated, False, info

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=noise_low, high=noise_high
        )
        qvel = self.init_qvel + self.np_random.uniform(
            size=self.model.nv, low=noise_low, high=noise_high
        )
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel]).ravel()
    
    def _get_reward(self, action):
        names = ['position_error', 'velocity_error', 'attitude_error', 'input', 'delta_acs']

        w_position  = 80.0 # 80.0
        w_velocity  = 20.0
        w_attitude  = 20.0
        w_input     = 10.0
        w_delta_acs = 10.0 #40.0 #3.0 #5.0

        reward_weights = np.array([w_position, w_velocity, w_attitude, w_input, w_delta_acs])
        weights = reward_weights / np.sum(reward_weights)  # weight can be adjusted later

        scale_pos       = 1.0
        scale_vel       = 1e-1
        scale_att       = 1e-1
        scale_input     = 5.0
        scale_delta_acs = 1e-1  # 2.0

        desired_pos = np.array([0.0, 0.0, 2.0]).reshape(3,1) # x y z 
        desired_vel = np.array([0.0, 0.0, 0.0]).reshape(3,1) # vx vy vz
        desired_att = np.array([0.0, 0.0, 0.0]).reshape(3,1) # roll, pitch, yaw
        current_pos = self.data.qpos
        current_vel = self.data.qvel
        current_att = quat2euler_raw(self.data.qpos[3:7]) # euler_mes
        
        pos_err = np.linalg.norm(current_pos - desired_pos) 
        r_pos = np.exp(-scale_pos * pos_err)

        vel_err = np.linalg.norm(current_vel- desired_vel) 
        r_vel = np.exp(-scale_vel * vel_err)  # scale_vel need to be adjust later

        att_err = np.linalg.norm(current_att- desired_att)
        r_att = np.exp(-scale_att * att_err)

        input_err = np.linalg.norm(action) 
        r_input = np.exp(-scale_input * input_err)

        delta_acs_err = np.linalg.norm(action - self.last_acs) 
        r_delta_acs = np.exp(-scale_delta_acs * delta_acs_err)

        rewards = np.array([r_pos, r_vel, r_att, r_input, r_delta_acs])
        total_reward = np.sum(weights * rewards)
        reward_dict = dict(zip(names, weights * rewards))

        return total_reward, reward_dict
    

if __name__ == "__main__":
    env = FlappyEnv(render_mode="human")
    for _ in range(10):
        action = env.action_space.sample()
        print(action)
    print(env.model.actuator_ctrlrange)