# Helpers
import os, sys
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(os.path.join(parent_dir))
sys.path.append(os.path.join(parent_dir, 'envs'))

import numpy as np
from typing import Dict, Union
from collections import deque
import matplotlib.pyplot as plt
import pandas as pd
# Gym
from gymnasium.spaces import Box
from gymnasium import utils
from gymnasium.utils import seeding
# Mujoco
import mujoco as mj
from mujoco_gym.mujoco_env import MujocoEnv
# Flappy
from dynamics import Flappy
from parameter import Simulation_Parameter
from aero_force import aero
from action_filter import ActionFilterButter
from env_randomize import EnvRandomizer
from utility_functions import *
import utility_trajectory as ut
from rotation_transformations import *
from pid_controller import PID_Controller
from fdc_dynamics import *
from symbolic_functions.func_joint_angles import *


DEFAULT_CAMERA_CONFIG = {"trackbodyid": 0, "distance": 10.0}
TRAJECTORY_TYPES = {"setpoint": 0, "linear": 1, "circular": 2}

class FlappyEnv(MujocoEnv, utils.EzPickle):
    metadata = {"render_modes": ["human", "rgb_array", "depth_array"]}
    
    def __init__(
        self,
        max_timesteps = 30000,
        xml_file: str = "../assets/Flappy_v8_FDC.xml",
        frame_skip: int = 1,
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        reset_noise_scale: float = 0.01,
        trajectory_type: str = "linear",
        env_num: int = 0,
        **kwargs
    ):
        self.model = mj.MjModel.from_xml_path(xml_file)
        self.data = mj.MjData(self.model)
        # NOTE: Parameters for Hard-Coded Version
        # region
        ##################################################
        #################### DYNAMICS ####################
        ##################################################
        self.p = Simulation_Parameter()
        self.sim = Flappy(p=self.p, render=False)
        self.sim_freq: int         = self.sim.freq # NOTE: 2000Hz for hard coding
        # self.dt                    = 1e-3 # NOTE: 1.0 / self.sim_freq = 1/2000s for hard coding
        self.frame_skip = frame_skip
        self.policy_freq: float    = 30.0 # NOTE: 30Hz but the real control frequency might not be exactly 30Hz because we round up the num_sims_per_env_step
        self.num_sims_per_env_step = int(self.sim_freq // self.policy_freq) # 2000//30 = 66
        self.secs_per_env_step     = self.num_sims_per_env_step / self.sim_freq # 66/2000 = 0.033s
        self.policy_freq: int      = int(1.0/self.secs_per_env_step) # 1000/33 = 30Hz
        self.num_step_per_sec: int = int(1.0/self.dt) # 1000
        # self.xa = np.zeros(3*self.p.n_Wagner)
        # endregion
        ##################################################
        ###################### TIME ######################
        ##################################################
        self.max_timesteps: int = max_timesteps
        self.timestep: int      = 0
        self.time_in_sec: float = 0.0 # Time
        ##################################################
        ################### TRAJECTORY ###################
        ##################################################
        self.trajectory_type = trajectory_type
        if self.trajectory_type == "setpoint":
            self.ref_traj           = ut.Setpoint(np.array([0,0,2]))
            self.traj_history_len   = 1
            self.future_traj        = deque(maxlen=self.traj_history_len) # future_traj = [x_{t+1},v_{t+1}, x_{t+4},v_{t+4}, x_{t+7},v_{t+7}]
        if self.trajectory_type == "linear":
            self.goal_pos           = np.array([10,0,2])
            self.ref_traj           = ut.SmoothTraj3(x0=np.array([0,0,2]), xf=self.goal_pos, tf=100)
            self.traj_history_len   = 3
            self.future_traj        = deque(maxlen=self.traj_history_len) # future_traj = [x_{t+1},v_{t+1}, x_{t+4},v_{t+4}, x_{t+7},v_{t+7}]
            self.bonus_point        = np.array([np.array([i,0,2]) for i in range(int(self.goal_pos[0]+1))])
        ##################################################
        #################### BOOLEANS ####################
        ##################################################
        self.is_traj            = True
        self.is_lpf_action      = False # Low Pass Filter
        self.is_visual          = False
        self.is_transfer        = False
        self.is_debug           = False
        self.is_randomize       = False
        self.is_noisy           = False
        self.is_random_dynamics = False # True to randomize dynamics
        self.is_plotting_joint  = False
        self.is_aero            = False
        self.is_launch_control  = False
        self.is_action_bound    = False
        self.is_rs_reward       = False # Rich-Sutton Reward
        self.is_io_history      = True
        self.is_pid             = False
        self.is_bonus           = (self.trajectory_type == 'linear')
        self.is_previous_bonus  = False
        ##################################################
        ################## OBSERVATION ###################
        ##################################################
        self.env_num            = env_num
        self.n_state            = 84 # NOTE: change to the number of states *we can measure*
        self.n_action           = 13 # NOTE: change to the number of action
        self.history_len_short  = 4 # NOTE: [o_{t-4}:o_{t}, a_{t-4}:a_{t}], o_{t} = [sensordata, θ_5, θ_6]
        self.history_len_long   = 10
        self.history_len        = self.history_len_short
        self.previous_obs       = deque(maxlen=self.history_len)
        self.previous_act       = deque(maxlen=self.history_len)
        self.last_act           = np.zeros(self.n_action)
        self.action_space       = self._set_action_space()
        # self.action_space       = Box(low=-100, high=100, shape=(self.n_action,))
        self.observation_space  = self._set_observation_space()
        self.num_episode        = 0
        self.previous_epi_len   = deque(maxlen=10); [self.previous_epi_len.append(0) for _ in range(10)]
        ##################################################
        ##################### BOUNDS #####################
        ##################################################
        # NOTE: Lower & upper bounds do not actually limit the actions output from MLP network, manually clip instead
        self.pos_lb = np.array([-20,-20,0.5]) # fight space dimensions: xyz(m)
        self.pos_ub = np.array([20,20,10])
        self.speed_bound = 100.0
        ##################################################
        ################### MUJOCOENV ####################
        ##################################################
        self.body_list = ["Base","L1","L2","L3","L4","L5","L6","L7","L1R","L2R","L3R","L4R","L5R","L6R","L7R"]
        self.joint_list = ['J1','J2','J3','J5','J6','J7','J10','J1R','J2R','J3R','J5R','J6R','J7R','J10R']
        self.bodyID_dic, self.jntID_dic, self.posID_dic, self.jvelID_dic = self.get_bodyIDs(self.body_list)
        self.jID_dic = self.get_jntIDs(self.joint_list)

        utils.EzPickle.__init__(self, xml_file, frame_skip, reset_noise_scale, **kwargs)
        MujocoEnv.__init__(self, xml_file, frame_skip, observation_space=self.observation_space, default_camera_config=default_camera_config, **kwargs)
        # HACK
        self.action_space = self._set_action_space()
        self.metadata = {
            "render_modes": ["human", "rgb_array", "depth_array"],
            "render_fps": int(np.round(1.0 / self.dt))
        }
        self.observation_structure = {
            "qpos": self.data.qpos.size,
            "qvel": self.data.qvel.size,
        }
        self._reset_noise_scale = reset_noise_scale
        ##################################################
        ###################### FDC #######################
        ##################################################
        # FDC Parameters
        self.L3b_init = 6.71/1000
        self.L5a_init = 11.18/1000
        self.L5b_init = 10/1000
        FDC_scale = np.array([[0.875862068965517, 3], # L3b
                              [0.875862068965517, 1.64827586206897], # L5a
                              [0.875862068965517, 1.16551724137931]]) # L5b
        FDC_range = np.array([self.L3b_init*FDC_scale[0, :], self.L5a_init*FDC_scale[1,:],self.L5b_init*FDC_scale[2,:]])
        self.FDC_range = FDC_range
        # Initialize Theta1 driving gear angle(deg)
        self._init_FDC_states()

        # Initial FDC control input
        # Normalized FDC control input; each in range [0,1]
        self.FDC_ctrl_temp = [(self.L3b_init - FDC_range[0,0]) / (FDC_range[0,1] - FDC_range[0,0]),
                              (self.L5a_init - FDC_range[1,0]) / (FDC_range[1,1] - FDC_range[1,0]),
                              (self.L5b_init - FDC_range[2,0]) / (FDC_range[2,1] - FDC_range[2,0])]
        self.FDC_ctrl = np.array([self.FDC_ctrl_temp, self.FDC_ctrl_temp])  # [[Left],[Right]] <=== Initial control input
        ##################################################
        ############## STATE INITIALIZATION ##############
        ##################################################
        self._init_action_filter()
        # self._init_env_randomizer() # NOTE: Take out dynamics randomization first 
        self._init_env()
        self.reset()
        ##################################################
        ################## PID CONTROL ###################
        ##################################################
        self.pid_controller = PID_Controller(self)
        self.last_pid_ctrl = np.zeros(self.n_action)

    @property
    def dt(self) -> float:
        return self.model.opt.timestep * self.frame_skip
    
    def _set_action_space(self):
        low = np.concatenate([np.zeros(12),[5]])
        high = np.concatenate([np.ones(12),[8]])
        if self.is_action_bound: self.action_space = Box(low=0.2*np.ones(12), high=0.8*np.ones(12))
        else: self.action_space = Box(low=low, high=high)
        return self.action_space

    def _set_observation_space(self):
        if self.is_io_history:
            if self.is_traj:
                # NOTE: obs = [o_t-4:o_t, a_t-4:a_t, o_t, future_traj] 
                # Setpoint: shape=(119,)=13x4+12x4+13x1+6, Linear: shape=(131,)=13x4+12x4+13x1+6x3
                # future_traj = [x_{t+1},v_{t+1}, x_{t+4},v_{t+4}, x_{t+7},v_{t+7}]
                obs_shape = (self.data.sensordata.shape[0])*(self.history_len+1) + self.action_space.shape[0]*self.history_len + (3+3)*self.traj_history_len
            else:
                # NOTE: obs = [o_t-4:o_t, a_t-4:a_t, o_t], shape=(107,)=15x4+8x4+15x1, o_t = [sensordata, joint_5,6]
                obs_shape = (self.data.sensordata.shape[0])*(self.history_len+1) + self.action_space.shape[0]*self.history_len
        else:
            obs_shape = self.data.sensordata.shape[0]
        observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_shape,))
        return observation_space

    def _init_env(self):
        print("Environment {} created".format(self.env_num))
        print("Sample action: {}".format(self.action_space.sample()))
        print("Action Space: {}".format(np.array(self.action_space)))
        print("Observation Space: {}".format(np.array(self.observation_space)))
        # print("Launch control: {}".format(self.is_launch_control))
        print("Time step(sec): {}".format(self.dt))
        # print("Policy frequency(Hz): {}".format(self.policy_freq))
        # print("Num sims / Env step: {}".format(self.num_sims_per_env_step))
        print("-"*100)

    def _init_FDC_states(self):
        self.theta1 = np.deg2rad(-90)
        self.d_theta1 = self.sim.flapping_freq*2*np.pi  # rad/s
        self.dd_theta1 = 0
        self.flapping_freq = self.sim.flapping_freq
        # Initialize FDC states
        self.FDC_pos = np.array([[self.L3b_init, self.L5a_init, self.L5b_init], [self.L3b_init, self.L5a_init, self.L5b_init]])
        self.FDC_vel = np.zeros_like(self.FDC_pos)

    def _init_action_filter(self):
        self.action_filter = ActionFilterButter(
            lowcut        = None,
            highcut       = [4],
            sampling_rate = self.policy_freq,
            order         = 2,
            num_joints    = self.n_action,
        )

    def reset(self, seed=None, randomize=None):
        super().reset(seed=self.env_num)
        if randomize is None: randomize = self.is_randomize
        self._reset_env(randomize)
        self.action_filter.reset()
        # self.env_randomizer.randomize_dynamics()
        # self._set_dynamics_properties()
        self._update_data(step=False, obs_curr=None, action=None)
        obs = self._get_obs()
        info = self._get_reset_info
        return obs, info
    
    def _reset_env(self, randomize=False):
        self.timestep    = 0 # discrete timestep, k
        self.time_in_sec = 0.0 # time
        self.reset_model()
        # use action
        self.last_act   = np.zeros(self.n_action)
        self.reward     = None
        self.terminated = None
        self.info       = {}

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

    def _init_env_randomizer(self):
        self.env_randomizer = EnvRandomizer(self.sim)

    def _set_dynamics_properties(self):
        if self.is_random_dynamics: self.sim.set_dynamics()

    def _get_obs(self):
        # NOTE: obs = [o_t-4:o_t, a_t-4:a_t, o_t, future_traj] 
        # Setpoint: shape=(119,)=13x4+12x4+13x1+6, Linear: shape=(131,)=13x4+12x4+13x1+6x3
        # future_traj = [x_{t+1},v_{t+1}, x_{t+4},v_{t+4}, x_{t+7},v_{t+7}]
        obs_curr = self.data.sensordata
        if self.is_io_history:
            if self.timestep == 0:
                [self.previous_obs.append(obs_curr) for _ in range(self.history_len)]
                [self.previous_act.append(np.zeros(self.n_action)) for _ in range(self.history_len)]
                if self.is_traj:
                    for i in range(self.traj_history_len):
                        desired_pos, desired_vel, desired_acc = self.ref_traj.get(self.time_in_sec+(3*i+1)*self.dt)
                        self.future_traj.append(np.concatenate([desired_pos, desired_vel]))
            obs_prev = np.concatenate([np.array(self.previous_obs,dtype=object).flatten(), np.array(self.previous_act,dtype=object).flatten()])
            future_traj = np.array(self.future_traj,dtype=object).flatten()
            obs = np.concatenate([obs_prev, obs_curr, future_traj])
        else:
            obs = obs_curr
        return obs

    def _act_norm2actual(self, act):
        return self.action_lower_bounds_actual + (act + 1)/2.0 * (self.action_upper_bounds_actual - self.action_lower_bounds_actual)

    def step(self, action, restore=False):
        # 1. Action Filter
        if self.timestep == 0: self.action_filter.init_history(action)
        if self.is_lpf_action: action_filtered = self.action_filter.filter(action)
        else: action_filtered = np.copy(action)
        # 2. Simulate for Single Time Step
        # for _ in range(self.num_sims_per_env_step):
        self.do_simulation(action_filtered, self.frame_skip) # a_{t}
        if self.render_mode == "human": self.render()
        # 3. Get Observation
        obs = self._get_obs() # o_{t+1}
        # PRINT OBS
        # region
        # print("Previous observation")
        # print(obs[0 : (self.data.sensordata.shape[0]+2)*self.history_len].reshape(4,15))
        # print("Previous action")
        # print(obs[(self.data.sensordata.shape[0]+2)*self.history_len : (self.data.sensordata.shape[0]+2+self.action_space.shape[0])*self.history_len].reshape(4,8))
        # print("Current observation")
        # print(obs[(self.data.sensordata.shape[0] + self.action_space.shape[0] + 2) * self.history_len : (self.data.sensordata.shape[0] + self.action_space.shape[0] + 2) * self.history_len + 15].reshape(1,15))
        # print("Future trajectory")
        # print(obs[(self.data.sensordata.shape[0] + self.action_space.shape[0] + 2) * self.history_len + 15 :].reshape(3,6))
        # endregion
        if self.is_io_history: obs_curr = obs[(self.data.sensordata.shape[0]+self.action_space.shape[0])*self.history_len : (self.data.sensordata.shape[0]+self.action_space.shape[0])*self.history_len+self.data.sensordata.shape[0]]
        else: obs_curr = obs
        # 4. Get Reward
        reward, reward_dict = self._get_reward(action_filtered, obs_curr)
        self.info["reward_dict"] = reward_dict
        # 5. Termination / Truncation
        terminated = self._terminated(obs_curr)
        # if self.is_rs_reward and (not self.is_transfer): reward += int(not terminated) * 0.1
        truncated = self._truncated()
        # 6. Update Data
        self._update_data(step=True, obs_curr=obs_curr, action=action_filtered)
        self.last_act = action_filtered
        # 7. ETC
        # if self.is_plotting_joint and self.timestep == 500: self._plot_joint() # Plot recorded data
        if terminated and self.timestep < (np.average(self.previous_epi_len)//1000+1)*1000:
            reward -= (10 - np.average(self.previous_epi_len)//1000) # Early Termination Penalty
        # if terminated:
        #     print("Episode terminated")
        #     print("Last action: {}".format(np.round(self.last_act[2:],2)))
        #     if self.is_pid:
        #         print("Last PID control: {}".format(np.round(self.last_pid_ctrl,2)))
        #         self.pid_controller.reset()
        #     print("Previous obs: {}".format(np.round(self.previous_obs,2)))
        #     print("Previous act: {}".format(np.round(self.previous_act,2)))

        return obs, reward, terminated, truncated, self.info
    
    def do_simulation(self, ctrl, n_frames) -> None:
        # if np.array(ctrl).shape != (self.model.nu,):
        #     raise ValueError(f"Action dimension mismatch. Expected {(self.model.nu,)}, found {np.array(ctrl).shape}")
        self._step_mujoco_simulation(ctrl, n_frames)

    def _step_mujoco_simulation(self, ctrl, n_frames):
        # NOTE: PID ONLY: 100Hz
        # pid_ctrl = self.pid_controller.control(self.data.sensordata)
        # if pid_ctrl is None:
        #     pid_ctrl = self.last_pid_ctrl
        # else:
        #     self.last_pid_ctrl = pid_ctrl
        # ctrl[2:] = pid_ctrl
        # print(np.round(pid_ctrl,2))

        # if self.is_launch_control and self.timestep < 1000: ctrl = self._launch_control(ctrl)
        self._apply_control(ctrl=ctrl)
        mj.mj_step(self.model, self.data, nstep=n_frames)

    def _launch_control(self, ctrl):
        # NOTE: Emperical
        # ctrl[2:6] = np.array([0.6,0.6,0.6,0.6])
        # ctrl[6:] = np.array([0.0,0.0])
        # NOTE: PID ONLY: 100Hz
        pid_ctrl = self.pid_controller.control(self.data.sensordata)
        if pid_ctrl is None:
            pid_ctrl = self.last_pid_ctrl
        else:
            self.last_pid_ctrl = pid_ctrl
        ctrl[2:] = pid_ctrl
        return ctrl

    def _apply_control(self, ctrl):
        self.data.actuator("Motor1").ctrl[0] = ctrl[0]  # data.ctrl[1] # front
        self.data.actuator("Motor2").ctrl[0] = ctrl[1]  # data.ctrl[2] # back
        self.data.actuator("Motor3").ctrl[0] = ctrl[2]  # data.ctrl[3] # left
        self.data.actuator("Motor4").ctrl[0] = ctrl[3]  # data.ctrl[4] # right
        self.data.actuator("Motor5").ctrl[0] = ctrl[4]  # data.ctrl[5] # left H
        self.data.actuator("Motor6").ctrl[0] = ctrl[5]  # data.ctrl[6] # right H

        # NOTE: If Using Custom Aero
        if self.is_aero:
            self.xd, R_body = self._get_original_states()
            fa, ua, self.xd = aero(self.model, self.data, self.xa, self.xd, R_body)
            # Apply Aero forces
            self.data.qfrc_applied[self.jvelID_dic["L3"]] = ua[0]
            self.data.qfrc_applied[self.jvelID_dic["L7"]] = ua[1]
            self.data.xfrc_applied[self.bodyID_dic["Base"]] = [*ua[2:5], *ua[5:8]]
            # Integrate Aero States
            self.xa = self.xa + fa * self.dt

        flap_freq = ctrl[12]
        self.flapping_freq = flap_freq # self.flapping_freq*(np.sin(np.pi*2*self.timestep)+1)  # <===== (some time-varying input just for checking; need to be replaced to some other value)
        # Integrate theta 1 (driving gear)
        d_theta1_ref = 2*np.pi*flap_freq
        kd_theta1 = 10

        # Compute theta1 dynamics using RK4 integration
        theta1_state = np.array([self.theta1, self.d_theta1])
        theta1_state = rk4(theta1_dynamics, theta1_state, d_theta1_ref, self.dt, kd_theta1)
        self.theta1, self.d_theta1 = theta1_state

        self.FDC_ctrl[0,0] = ctrl[6] # np.sin(self.sim.flapping_freq * self.timestep)  # Left L3b
        self.FDC_ctrl[0,1] = ctrl[7] # np.sin(self.sim.flapping_freq * self.timestep)  # Left L5a
        self.FDC_ctrl[0,2] = ctrl[8] # np.sin(self.sim.flapping_freq * self.timestep)  # Left L5b
        self.FDC_ctrl[1,0] = ctrl[9] # np.sin(self.sim.flapping_freq * self.timestep)  # Left L3a
        self.FDC_ctrl[1,1] = ctrl[10] # np.sin(self.sim.flapping_freq * self.timestep)  # Left L5a
        self.FDC_ctrl[1,2] = ctrl[11] # np.sin(self.sim.flapping_freq * self.timestep)  # Left L5b

        FDC_ref = np.array([FDC_ctrlInput2Length(self.FDC_ctrl[0,:], self.FDC_range),
                            FDC_ctrlInput2Length(self.FDC_ctrl[1,:], self.FDC_range)])
        # FDC Dynamics
        kp_fdc = 10000
        kd_fdc = 200
        # Compute FDC dynamics using RK4 integration
        FDC_state = np.array([self.FDC_pos, self.FDC_vel])
        FDC_state = rk4(fdc_dynamics, FDC_state, FDC_ref, self.dt, kp_fdc, kd_fdc)
        self.FDC_pos, self.FDC_vel = FDC_state

        # Clamp FDC length
        self.FDC_pos = clampFDC(self.FDC_pos, self.FDC_range)
        # Compute joint angles
        J5_l, J6_l = FDC_length2JointAngle(self.FDC_pos[0,:], wrap_angle(self.theta1))  # Left
        J5_r, J6_r = FDC_length2JointAngle(self.FDC_pos[1,:], wrap_angle(self.theta1))  # Right
        # Apply angles to Joints
        self.data.actuator("J5_angle").ctrl[0] = J5_l
        self.data.actuator("J6_angle").ctrl[0] = J6_l
        self.data.actuator("J5R_angle").ctrl[0] = J5_r
        self.data.actuator("J6R_angle").ctrl[0] = J6_r

        # Update wing_conformation
        self.p.wing_conformation_L[5] = self.FDC_pos[0, 0]
        self.p.wing_conformation_L[9] = self.FDC_pos[0, 1]
        self.p.wing_conformation_L[10] = self.FDC_pos[0, 2]
        self.p.wing_conformation_R[5] = self.FDC_pos[1, 0]
        self.p.wing_conformation_R[9] = self.FDC_pos[1, 1]
        self.p.wing_conformation_R[10] = self.FDC_pos[1, 2]

        # Compute actual J6v from finite difference
        if self.data.time == 0:
            J6v_fd = [0, 0]
            self.J6_old = np.array([self.J6_l, self.J6_r])
        J6_current = np.array([self.J6_l, self.J6_r])
        J6v_fd = (J6_current - self.J6_old) / self.dt
        self.J6_old = J6_current  # update

        # NOTE: If Using Custom Aero
        if self.is_aero: #self aero not implemented yet?
            xd_L, xd_R, R_body = self._get_original_states()

            # Update Joint 6 velocity with finite difference for aero force computation
            xd_L[6] = J6v_fd[0]
            xd_R[6] = J6v_fd[1]

            fa, ua = aero(self.model, self.data, self.xa, xd_L, xd_R, R_body, self.p)

            # Apply Aero forces
            self.data.xfrc_applied[self.bodyID_dic["Base"]] = [*ua[2:5], *ua[5:8]]

            # Integrate Aero States
            self.xa = self.xa + fa * self.dt

    # NOTE: For aero()
    def _get_original_states(self):
        xd = np.array([0.0] * 22)
        xd[0] = self.data.qpos[self.posID_dic["L3"]] + np.deg2rad(11.345825599281223)  # xd[0:1] left wing angles [shoulder, elbow]
        xd[1] = self.data.qpos[self.posID_dic["L7"]] - np.deg2rad(27.45260202) + xd[0]
        xd[2:5] = self.data.sensordata[0:3]  # inertial position (x, y, z)
        xd[5] = self.data.qvel[self.jvelID_dic["L3"]]  # Joint 5 velocity
        xd[6] = self.data.qvel[self.jvelID_dic["L7"]]  # Joint 6 velocity
        xd[7:10] = self.data.sensordata[7:10]  # Inertial Frame Linear Velocity
        xd[10:13] = self.data.sensordata[10:13]  # Body Frame Angular velocity

        if np.linalg.norm(self.data.sensordata[3:7]) == 0:
            self.data.sensordata[3:7] = [1,0,0,0]
        R_body = quat2rot(self.data.sensordata[3:7])
        xd[13:23] = list(np.transpose(R_body).flatten())
        return xd, R_body

    def _update_data(self, obs_curr, action, step=True):
        if step:
            # Past
            self.previous_obs.append(obs_curr)
            self.previous_act.append(action)
            # Now
            self.time_in_sec = np.round(self.time_in_sec + self.dt, 3)
            self.timestep += 1
            # Future
            desired_pos, desired_vel, desired_acc = self.ref_traj.get(self.time_in_sec+(3*2+1)*self.dt)
            self.future_traj.append(np.concatenate([desired_pos, desired_vel]))

    def _get_reward(self, action, obs_curr):
        names = ['pos_rew', 'vel_rew', 'ang_vel_rew', 'att_rew', 'input_rew', 'delta_act_rew', 'pid_rew', 'wing_dist_rew']

        w_position         = np.average(self.previous_epi_len)//1000 + 1 # Focus on position more as it can fly better
        w_velocity         = np.average(self.previous_epi_len)//2000 + 1
        w_angular_velocity = 10.0
        w_attitude         = 10.0
        w_input            = 0.5
        w_delta_act        = 0.5
        w_pid              = 1.0
        w_wing_dist        = 10.0

        reward_weights = np.array([w_position, w_velocity, w_angular_velocity, w_attitude, w_input, w_delta_act, w_pid])
        weights = reward_weights / np.sum(reward_weights)  # weight can be adjusted later

        scale_pos       = 1.0/5.0
        scale_att       = 1.0/(np.pi/2)
        scale_vel       = 1.0/5.0
        scale_ang_vel   = 1.0/2.0
        scale_input     = 1.0/15.0
        scale_delta_act = 1.0/1.0
        scale_pid       = 1.0/1.0
        scale_wing_dist = 1.0/0.01

        desired_pos, desired_vel, desired_acc = self.ref_traj.get(self.time_in_sec)

        # desired_pos     = np.array([0.0,0.0,2.0]) # x y z 
        # desired_ori     = np.array([1.0,0.0,0.0,0.0]) # roll, pitch, yaw
        desired_att     = quat2euler_raw(np.array([1.0,0.0,0.0,0.0])) # roll, pitch, yaw
        # desired_vel     = np.array([0.0,0.0,0.0]) # vx vy vz
        desired_ang_vel = np.array([0.0,0.0,0.0]) # \omega_x \omega_y \omega_z
        
        current_pos     = obs_curr[0:3]
        # current_ori     = obs_curr[3:7]
        current_att     = quat2euler_raw(obs_curr[3:7])
        current_vel     = obs_curr[7:10]
        current_ang_vel = obs_curr[10:13]

        pos_err       = np.linalg.norm(current_pos - desired_pos)
        # if self.is_transfer: pos_err += (np.linalg.norm(current_pos[0:2] - desired_pos[0:2]) + np.abs(current_pos[2]-desired_pos[2])) # Extra penalize pos error
        # ori_err       = np.linalg.norm(current_ori - desired_ori)
        att_err       = np.linalg.norm(np.array([1.0,0.5,1.0]) * (current_att - desired_att))
        vel_err       = np.linalg.norm(current_vel - desired_vel)
        ang_vel_err   = np.linalg.norm(np.array([1.0,0.5,1.0]) * (current_ang_vel - desired_ang_vel))
        input_err     = np.linalg.norm(action[:12]) # It's not an error but let's just call it
        delta_act_err = np.linalg.norm(action[:12] - self.last_act[:12]) # It's not an error but let's just call it
        pid_err       = self._get_pid_error(action[2:])
        wing_dist_err = self._get_wing_dist_error()

        rewards = np.exp(-np.array([scale_pos, scale_att, scale_vel, scale_ang_vel, scale_input, scale_delta_act, scale_pid, scale_wing_dist]
                         * np.array([pos_err, att_err, vel_err, ang_vel_err, input_err, delta_act_err, pid_err, wing_dist_err])))
        reward_dict = dict(zip(names, weights * rewards)) 
        total_reward = np.sum(weights * rewards)

        # NOTE: Position-based discontinuous bonus (Linear trajectory only)
        # if |x - 1| < 0.1: bonus = 0.1 * (1 - exp(-1)) = 0.063 / 0.086 / 0.095 / 0.098 / 0.099
        # bonus \in [0,0.1) and concave
        if self.is_bonus and self.trajectory_type == "linear":
            goal_pos_x = int(self.goal_pos[0])
            bonus = 0.1*(1-np.exp(-np.min([np.abs(round(current_pos[0])), goal_pos_x])))
            # In the bonus region
            if np.linalg.norm(current_pos - self.bonus_point[np.min([np.abs(round(current_pos[0])), goal_pos_x])]) < np.exp(-np.average(self.previous_epi_len)/10000):
                total_reward += bonus
                # Print only when it just entered the bonus region
                if (not self.is_previous_bonus) and bonus > 0:
                    print("Env {env_num}  |  Bonus started  |  Bonus Point: {bp}  |  Postion: {pos}  |  Bonus: {bonus}".format(
                          env_num=self.env_num,
                          bp=self.bonus_point[np.min([np.abs(round(current_pos[0])), goal_pos_x])],
                          pos=np.round(np.array(current_pos, dtype=float), 2),
                          bonus=np.round(bonus, 3)))
                    self.is_previous_bonus = True
            # Out of the bonus region
            # Print only when it just came out of the bonus region
            elif self.is_previous_bonus:
                print("Env {env_num}  |  Bonus terminated  |  Postion: {pos}".format(
                        env_num=self.env_num,
                        pos=np.round(np.array(current_pos, dtype=float), 2)))
                self.is_previous_bonus = False
            else:
                self.is_previous_bonus = False
        
        return total_reward, reward_dict

    def _get_pid_error(self, action):
        if self.is_pid:
            pid_ctrl = self.pid_controller.control(self.previous_obs[-1]) # PID ctrl for the past obs
            if pid_ctrl is None:
                pid_err = np.linalg.norm(action - self.last_pid_ctrl)
            else:
                pid_err = np.linalg.norm(action - pid_ctrl)
                self.last_pid_ctrl = pid_ctrl
        else:
            pid_err = 0
        return pid_err

    def _get_wing_dist_error(self):
        wing_dist = self.data.contact.dist
        if wing_dist < 0.1: return wing_dist
        else: return 0 

    def _terminated(self, obs_curr):
        pos = np.array(obs_curr[0:3], dtype=float)
        vel = np.array(obs_curr[7:10], dtype=float)
        att = quat2euler_raw(obs_curr[3:7]) # roll, pitch, yaw
        if not((pos <= self.pos_ub).all() 
           and (pos >= self.pos_lb).all()):
            self.num_episode += 1
            self.previous_epi_len.append(self.timestep)
            print("Env {env_num}  |  Episode {epi}  |  Out of position bounds: {pos}  |  Timestep: {timestep}  |  Time: {time}s".format(
                  env_num=self.env_num,
                  epi=self.num_episode,
                  pos=np.round(pos,2),
                  timestep=self.timestep,
                  time=round(self.timestep*self.dt,2)))
            return True
        elif not(np.linalg.norm(vel) <= self.speed_bound):
            self.num_episode += 1
            self.previous_epi_len.append(self.timestep)
            print("Env {env_num}  |  Episode {epi}  |  Out of speed bounds: {vel}  |  Timestep: {timestep}  |  Time: {time}s".format(
                  env_num=self.env_num,
                  epi=self.num_episode,
                  vel=np.round(vel,2),
                  timestep=self.timestep,
                  time=round(self.timestep*self.dt,2)))
            return True
        elif np.abs(att[0]) >= np.pi/2 or np.abs(att[1]) >= np.pi/2:
            print("Env {env_num}  |  Episode {epi}  |  Out of attitude bounds: {att}  |  Timestep: {timestep}  |  Time: {time}s".format(
                  env_num=self.env_num,
                  epi=self.num_episode,
                  att=np.round(att,2),
                  timestep=self.timestep,
                  time=round(self.timestep*self.dt,2)))
            return True
        elif self._get_wing_dist()[1]:
            print("Env {env_num}  |  Episode {epi}  |  Wing Collision: {col}  |  Timestep: {timestep}  |  Time: {time}s".format(
                  env_num=self.env_num,
                  epi=self.num_episode,
                  col=self._get_wing_dist()[0],
                  timestep=self.timestep,
                  time=round(self.timestep*self.dt,2)))
            return True
        elif self.timestep >= self.max_timesteps:
            print("Env {env_num}  |  Max step reached: Timestep: {timestep}  |  Position: {pos}  |  Time: {time}s".format(
                  env_num=self.env_num,
                  timestep=self.max_timesteps,
                  pos=np.round(pos,2),
                  time=round(self.timestep*self.dt,2)))
            return True
        else:
            return False

    def _truncated(self):
        return False

    def _get_wing_dist(self):
        if self.data.geom('Wing').id in self.data.contact.geom and self.data.geom('WingR').id in self.data.contact.geom:
            collide_val = self.data.contact.dist
            return collide_val, True
        return 100, False

    def get_bodyIDs(self, body_list):
        bodyID_dic = {}
        jntID_dic = {}
        posID_dic = {}
        jvelID_dic = {}
        for bodyName in body_list:
            mjID = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, bodyName)
            jntID = self.model.body_jntadr[mjID]   # joint ID
            jvelID = self.model.body_dofadr[mjID]  # joint velocity
            posID = self.model.jnt_qposadr[jntID]  # joint position
            bodyID_dic[bodyName] = mjID
            jntID_dic[bodyName] = jntID
            posID_dic[bodyName] = posID
            jvelID_dic[bodyName] = jvelID
        return bodyID_dic, jntID_dic, posID_dic, jvelID_dic

    def get_jntIDs(self, jnt_list):
        jointID_dic = {}
        for jointName in jnt_list:
            jointID = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, jointName)
            jointID_dic[jointName] = jointID
        return jointID_dic

    def close(self):
        if self.mujoco_renderer is not None:
            self.mujoco_renderer.close()

