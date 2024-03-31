# Helpers
import os, sys
sys.path.append('/Users/mintaekim/Desktop/HRL/Flappy/Integrated/Flappy_Integrated/flappy_v2')
sys.path.append('/Users/mintaekim/Desktop/HRL/Flappy/Integrated/Flappy_Integrated/flappy_v2/envs')
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


DEFAULT_CAMERA_CONFIG = {"trackbodyid": 0, "distance": 10.0}
TRAJECTORY_TYPES = {"linear": 0, "circular": 1, "setpoint": 2}

class FlappyEnv(MujocoEnv, utils.EzPickle):
    metadata = {"render_modes": ["human", "rgb_array", "depth_array"]}
    
    def __init__(
        self,
        max_timesteps = 20000,
        # xml_file: str = "../assets/Flappy_v8_FixedAxis.xml",
        xml_file: str = "../assets/Flappy_v8_JointInput.xml",
        # xml_file: str = "../assets/Flappy_v8_Base.xml",
        frame_skip: int = 1,
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        reset_noise_scale: float = 0.01,
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
        self.num_step_per_sec      = int(1.0/self.dt) # 1000
        # self.xa = np.zeros(3*self.p.n_Wagner)
        # endregion
        ##################################################
        ###################### TIME ######################
        ##################################################
        self.max_timesteps      = max_timesteps
        self.timestep: int      = 0
        self.time_in_sec: float = 0.0 # Time
        ##################################################
        ################### TRAJECTORY ###################
        ##################################################
        self.ref_traj           = ut.SmoothTraj5(x0=np.array([0,0,2]), xf=np.array([5,0,2]), tf=10)
        self.traj_history_len   = 3
        self.future_traj        = deque(maxlen=self.traj_history_len) # future_traj = [x_{t+1},v_{t+1}, x_{t+4},v_{t+4}, x_{t+7},v_{t+7}]
        
        ##################################################
        #################### Booleans ####################
        ##################################################
        self.is_traj            = True
        self.is_lpf_action      = True # Low Pass Filter
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
        ##################################################
        ################## Observation ###################
        ##################################################
        self.n_state            = 84 # NOTE: change to the number of states *we can measure*
        self.n_action           = 8 # NOTE: change to the number of action
        self.history_len_short  = 4 # NOTE: [o_{t-4}:o_{t}, a_{t-4}:a_{t}], o_{t} = [sensordata, θ_5, θ_6]
        self.history_len_long   = 10
        self.history_len        = self.history_len_short
        self.previous_obs       = deque(maxlen=self.history_len)
        self.previous_act       = deque(maxlen=self.history_len)
        self.last_act           = np.zeros(self.n_action)
        self.action_space       = self._set_action_space()
        self.observation_space  = self._set_observation_space()
        self.num_episode        = 0
        self.previous_epi_len   = deque(maxlen=10); [self.previous_epi_len.append(0) for _ in range(10)]
        ##################################################
        ##################### BOUNDS #####################
        ##################################################
        # NOTE: Lower & upper bounds do not actually limit the actions output from MLP network, manually clip instead
        self.pos_lb = np.array([-20,-20,0.5]) # fight space dimensions: xyz(m)
        self.pos_ub = np.array([20,20,10])
        self.speed_bound = 10.0
        ##################################################
        ################### MujocoEnv ####################
        ##################################################
        self.body_list = ["Base","L1","L2","L3","L4","L5","L6","L7","L1R","L2R","L3R","L4R","L5R","L6R","L7R"]
        self.joint_list = ['J1','J2','J3','J5','J6','J7','J10','J1R','J2R','J3R','J5R','J6R','J7R','J10R']
        self.bodyID_dic, self.jntID_dic, self.posID_dic, self.jvelID_dic = self.get_bodyIDs(self.body_list)
        self.jID_dic = self.get_jntIDs(self.joint_list)
        ##################################################
        ################## Joint Input ###################
        ##################################################
        self.Angle_data = pd.read_csv("../assets/JointAngleData.csv", header=None)
        self.J5_m = self.Angle_data.loc[:,0]  # Mujoco reference joint angle (θ_5)
        self.J6_m = self.Angle_data.loc[:,1]  # Mujoco reference joint angle (θ_6)
        self.J5v_m = self.sim.flapping_freq/2 * self.Angle_data.loc[:,2]  # Mujoco reference joint angle velocity (θ_5_dot)
        self.J6v_m = self.sim.flapping_freq/2 * self.Angle_data.loc[:,3]  # Mujoco reference joint angle velocity (θ_6_dot)
        self.t_m = np.linspace(0, 1.0/self.sim.flapping_freq, num=len(self.J5_m))
        self.joint_5_6 = np.zeros(2)
        ##################################################
        ######## Record Structure for Early Stage ########
        ##################################################
        self.SimTime = []
        self.ua_ = []
        self.JointAng = [[],[]]
        self.JointAng_ref = [[],[]]
        self.JointVel = [[],[]]
        self.JointVel_ref = [[],[]]

        utils.EzPickle.__init__(self, xml_file, frame_skip, reset_noise_scale, **kwargs)
        MujocoEnv.__init__(self, xml_file, frame_skip, observation_space=self.observation_space, default_camera_config=default_camera_config, **kwargs)
        
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
        ############## STATE NORMALIZATION ###############
        ##################################################
        self._init_action_filter()
        # self._init_env_randomizer() # NOTE: Take out dynamics randomization first 
        self._seed()
        self.reset()
        self._init_env()
        ##################################################
        ################## PID CONTROL ###################
        ##################################################
        self.pid_controller = PID_Controller(self)
        self.last_pid_ctrl = np.zeros(self.n_action-2)

    @property
    def dt(self) -> float:
        return self.model.opt.timestep * self.frame_skip

    def _set_action_space(self):
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        if self.is_action_bound:
            self.action_space = Box(low=np.array([-2.0, 1.57, 0.4, 0.4, 0.4, 0.4, 0.0, 0.0]),
                                    high=np.array([-1.57, 1.57, 0.8, 0.8, 0.8, 0.8, 0.3, 0.3]))
        else:
            self.action_space = Box(low=low, high=high)
        return self.action_space

    def _set_observation_space(self):
        if self.is_io_history:
            if self.is_traj:
                # NOTE: obs = [o_t-4:o_t, a_t-4:a_t, o_t, future_traj], shape=(125,)=15x4+8x4+15x1+6*3
                # o_t = [sensordata, joint_5,6]
                # future_traj = [x_{t+1},v_{t+1}, x_{t+4},v_{t+4}, x_{t+7},v_{t+7}]
                obs_shape = (self.data.sensordata.shape[0]+2)*(self.history_len+1) + self.action_space.shape[0]*self.history_len + (3+3)*self.traj_history_len
            else:
                # NOTE: obs = [o_t-4:o_t, a_t-4:a_t, o_t], shape=(107,)=15x4+8x4+15x1, o_t = [sensordata, joint_5,6]
                obs_shape = (self.data.sensordata.shape[0]+2)*(self.history_len+1) + self.action_space.shape[0]*self.history_len
        else:
            obs_shape = self.data.sensordata.shape[0]+2
        observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_shape,))
        return observation_space

    def _init_env(self):
        print("Environment created")
        print("Sample action: {}".format(self.action_space.sample()))
        print("Action Space: {}".format(np.array(self.model.actuator_ctrlrange.T)))
        print("Actual Action Space: {}".format(np.array(self.action_space)))
        print("Observation Space: {}".format(np.array(self.observation_space)))
        # print("Launch control: {}".format(self.is_launch_control))
        print("Time step(sec): {}".format(self.dt))
        # print("Policy frequency(Hz): {}".format(self.policy_freq))
        # print("Num sims / Env step: {}".format(self.num_sims_per_env_step))
        print("-"*100)

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
        if randomize is None:
            randomize = self.is_randomize
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
        if self.is_random_dynamics:
            self.sim.set_dynamics()

    def _get_obs(self):
        # NOTE: obs = [o_t-4:o_t, a_t-4:a_t, o_t], shape=(107,)=15x4+8x4+15x1
        obs_curr = np.concatenate([self.data.sensordata, self.joint_5_6]) # shape = 13 + 2
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
        # 1. Action Filer
        if self.timestep == 0: self.action_filter.init_history(action)
        if self.is_lpf_action: action_filtered = self.action_filter.filter(action)
        else: action_filtered = np.copy(action)
        # 2. Simulate for Single Time Step
        # for _ in range(self.num_sims_per_env_step):
        self.do_simulation(action_filtered, self.frame_skip) # a_{t}
        if self.render_mode == "human": self.render()
        # 3. Get Observation
        obs = self._get_obs() # o_{t+1}
        print(obs)
        if self.is_io_history: obs_curr = obs[(self.data.sensordata.shape[0] + self.action_space.shape[0] + 2) * self.history_len :]
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
        if terminated:
            print("Episode terminated")
            print("Last action: {}".format(np.round(self.last_act[2:],2)))
            if self.is_pid:
                print("Last PID control: {}".format(np.round(self.last_pid_ctrl,2)))
                self.pid_controller.reset()
            # print("Previous obs: {}".format(np.round(self.previous_obs,2)))
            # print("Previous act: {}".format(np.round(self.previous_act,2)))

        return obs, reward, terminated, truncated, self.info
    
    def do_simulation(self, ctrl, n_frames) -> None:
        if np.array(ctrl).shape != (self.model.nu,):
            raise ValueError(f"Action dimension mismatch. Expected {(self.model.nu,)}, found {np.array(ctrl).shape}")
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

        if self.is_launch_control and self.timestep < 1000: ctrl = self._launch_control(ctrl)
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
        self.data.actuator("Motor1").ctrl[0] = ctrl[2]  # data.ctrl[1] # front
        self.data.actuator("Motor2").ctrl[0] = ctrl[3]  # data.ctrl[2] # back
        self.data.actuator("Motor3").ctrl[0] = ctrl[4]  # data.ctrl[3] # left
        self.data.actuator("Motor4").ctrl[0] = ctrl[5]  # data.ctrl[4] # right
        self.data.actuator("Motor5").ctrl[0] = ctrl[6]  # data.ctrl[5] # left H
        self.data.actuator("Motor6").ctrl[0] = ctrl[7]  # data.ctrl[6] # right H

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

        # Joint Input Data
        _J5 = np.interp(self.timestep*self.dt, self.t_m, self.J5_m, period=1.0/self.sim.flapping_freq)
        _J6 = np.interp(self.timestep*self.dt, self.t_m, self.J6_m, period=1.0/self.sim.flapping_freq)
        J5_d = _J5 - np.deg2rad(11.345825599281223)  # convert to Mujoco Reference
        J6_d = _J6 + np.deg2rad(27.45260202) - _J5
        # Apply angles to Joints
        # J5_d, J6_d = 0, 0
        self.data.actuator("J5_angle").ctrl[0] = J5_d
        self.data.actuator("J6_angle").ctrl[0] = J6_d
        # NOTE: Record joint angles
        if self.timestep < 500: self._record_joint(_J5=_J5, _J6=_J6)
        self.joint_5_6 = np.array([_J5, _J6])

    def _record_joint(self, _J5, _J6):
        J5 = self.data.qpos[self.posID_dic["L3"]] + np.deg2rad(11.345825599281223) # Get angle from mujoco
        J6 = self.data.qpos[self.posID_dic["L7"]] - np.deg2rad(27.45260202) + J5
        self.SimTime.append(np.round((self.timestep + 1) * self.dt, 3))
        self.JointAng_ref[0].append(_J5)
        self.JointAng_ref[1].append(_J6)
        J5v_d = np.interp(np.round(self.data.time,3), self.t_m, self.J5v_m, period=1.0 / self.sim.flapping_freq) # Get velocity just for comparison
        J6v_d = np.interp(np.round(self.data.time,3), self.t_m, self.J6v_m, period=1.0 / self.sim.flapping_freq)
        self.JointVel_ref[0].append(J5v_d)
        self.JointVel_ref[1].append(J6v_d)
        self.JointAng[0].append(J5)
        self.JointAng[1].append(J6)
        self.JointVel[0].append(self.data.qvel[self.jvelID_dic["L3"]])
        self.JointVel[1].append(self.data.qvel[self.jvelID_dic["L7"]])

    def _plot_joint(self):
        plt.figure()
        plt.subplot(2,1,1)
        plt.title("Joint Angle (MuJoCo vs. Matlab)")
        plt.plot(self.SimTime[100:], np.rad2deg(self.JointAng_ref[0][100:]), 'b--', label='J5 ref')
        plt.plot(self.SimTime[100:], np.rad2deg(self.JointAng[0][100:]), 'b', label='J5 real')
        plt.xlabel('Time, t (s)')
        plt.ylabel('J5 Angle, θ_5 (deg)')
        plt.legend()

        plt.subplot(2,1,2)
        plt.plot(self.SimTime[100:], np.rad2deg(self.JointAng_ref[1][100:]), 'g--', label='J6 ref')
        plt.plot(self.SimTime[100:], np.rad2deg(self.JointAng[1][100:]), 'g', label='J6 real')
        plt.xlabel('Time, t (s)')
        plt.ylabel('J6 Angle, θ_6 (deg)')
        plt.legend()

        plt.figure()
        plt.subplot(2,1,1)
        plt.title("Joint Angle Velocity (MuJoCo vs. Matlab)")
        plt.plot(self.SimTime[100:], np.rad2deg(self.JointVel_ref[0][100:]), 'b--', label='J5 ref')
        plt.plot(self.SimTime[100:], np.rad2deg(self.JointVel[0][100:]), 'b', label='J5 real')
        plt.xlabel('Time, t (s)')
        plt.ylabel('J5 Speed, θ_5 (deg/s)')
        # plt.ylim([-2000,2000])
        plt.legend()

        plt.subplot(2,1,2)
        plt.plot(self.SimTime[100:], np.rad2deg(self.JointVel_ref[1][100:]), 'g--', label='J6 ref')
        plt.plot(self.SimTime[100:], np.rad2deg(self.JointVel[1][100:]), 'g', label='J6 real')
        plt.xlabel('Time, t (s)')
        plt.ylabel('J6 Speed, θ_6 (deg/s)')
        # plt.ylim([-3000,3000])
        plt.legend()
        plt.show()

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
        names = ['position_error', 'velocity_error', 'angular_velocity', 'orientation_error', 'input', 'delta_acs']

        w_position         = np.average(self.previous_epi_len)//1000 + 1 # Focus on position more as it can fly better
        w_velocity         = np.average(self.previous_epi_len)//2000 + 1
        w_angular_velocity = 10.0
        w_orientation      = 10.0
        w_input            = 0.5
        w_delta_act        = 0.5
        w_pid              = 1.0

        reward_weights = np.array([w_position, w_velocity, w_angular_velocity, w_orientation, w_input, w_delta_act, w_pid])
        weights = reward_weights / np.sum(reward_weights)  # weight can be adjusted later

        scale_pos       = 5.0
        scale_ori       = 1.0
        scale_vel       = 5.0
        scale_ang_vel   = 2.0
        scale_input     = 6.0
        scale_delta_act = 1.0
        scale_pid       = 1.0

        desired_pos, desired_vel, desired_acc = self.ref_traj.get(self.time_in_sec)

        # desired_pos     = np.array([0.0,0.0,2.0]) # x y z 
        desired_ori     = np.array([1.0,0.0,0.0,0.0]) # roll, pitch, yaw
        # desired_vel     = np.array([0.0,0.0,0.0]) # vx vy vz
        desired_ang_vel = np.array([0.0,0.0,0.0]) # \omega_x \omega_y \omega_z
        
        current_pos     = obs_curr[0:3]
        current_ori     = obs_curr[3:7]
        current_vel     = obs_curr[7:10]
        current_ang_vel = obs_curr[10:13]

        pos_err       = np.linalg.norm(current_pos - desired_pos)
        # if self.is_transfer: pos_err += (np.linalg.norm(current_pos[0:2] - desired_pos[0:2]) + np.abs(current_pos[2]-desired_pos[2])) # Extra penalize pos error
        ori_err       = np.linalg.norm(current_ori - desired_ori)
        vel_err       = np.linalg.norm(current_vel - desired_vel)
        ang_vel_err   = np.linalg.norm(current_ang_vel - desired_ang_vel)
        input_err     = np.linalg.norm(action[2:]) # It's not an error but let's just call it
        delta_act_err = np.linalg.norm(action[2:] - self.last_act[2:]) # It's not an error but let's just call it
        pid_err       = self._get_pid_error(action[2:])

        rewards = np.exp(-np.array([scale_pos, scale_ori, scale_vel, scale_ang_vel, scale_input, scale_delta_act, scale_pid]
                         * np.array([pos_err, ori_err, vel_err, ang_vel_err, input_err, delta_act_err, pid_err])))
        total_reward = np.sum(weights * rewards)
        reward_dict = dict(zip(names, weights * rewards))

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

    def _terminated(self, obs_curr):
        pos = np.array(obs_curr[0:3], dtype=float)
        vel = np.array(obs_curr[7:10], dtype=float)
        if not((pos <= self.pos_ub).all() 
           and (pos >= self.pos_lb).all()):
            self.num_episode += 1
            self.previous_epi_len.append(self.timestep)
            print("Episode {epi}  |  Out of position bounds: {pos}  |  Timestep: {timestep}  |  Time: {time}s".format(epi=self.num_episode, pos=np.round(pos,2), timestep=self.timestep, time=round(self.timestep*self.dt,2)))
            return True
        elif not(np.linalg.norm(vel) <= self.speed_bound):
            self.num_episode += 1
            self.previous_epi_len.append(self.timestep)
            print("Episode {epi}  |  Out of speed bounds: {vel}  |  Timestep: {timestep}  |  Time: {time}s".format(epi=self.num_episode, vel=np.round(vel,2), timestep=self.timestep, time=round(self.timestep*self.dt,2)))
            return True
        else:
            return False

    def _truncated(self):
        if self.timestep >= self.max_timesteps:
            pos = np.array(self.data.sensordata[0:3], dtype=float)
            print("Max step reached: Timestep: {timestep}  |  Position: {pos}  |  Time: {time}s".format(timestep=self.max_timesteps, pos=np.round(pos,2), time=round(self.timestep*self.dt,2)))
            return True
        else:
            return False

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