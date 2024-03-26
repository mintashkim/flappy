import numpy as np
from rotation_transformations import *

''' PID Linear Trajectory Tracking control; Track by translation only'''

class PID_Controller:
    def __init__(self, env, points=None, tspans=None):
        self.env = env
        self.model = self.env.model
        self.data = self.env.data
        # initialize states
        self.x_d = 0
        self.y_d = 0
        self.z_d = 2.0
        self.vx_d = 0
        self.vy_d = 0
        self.roll_d = 0
        self.pitch_d = 0
        self.yaw_d = 0  # np.pi/18
        self.roll_rate_d = 0
        self.pitch_rate_d = 0
        self.yaw_rate_d = 0

        self.pitch_I = 0
        self.roll_I = 0
        self.yaw_I = 0
        self.vx_I = 0
        self.vy_I = 0
        self.z_I = 0

        self.pitch_err_prv = 0
        self.roll_err_prv = 0
        self.yaw_err_prv = 0
        self.vx_err_prv = 0
        self.vy_err_prv = 0
        self.z_err_prv = 0

        self.time_ctrlPos_prv = self.env.timestep * self.env.dt
        self.time_ctrlAtt_prv = self.env.timestep * self.env.dt

        # Position Control
        self.Kp_Xh = 1
        self.Kp_vh = 1

        self.points = points
        self.tspans = tspans

    def reset(self):
        self.time_ctrlPos_prv = self.env.timestep * self.env.dt
        self.time_ctrlAtt_prv = self.env.timestep * self.env.dt

    def control(self, obs):
        # t = self.env.timestep * self.env.dt
        # pos_d = trajectory_generator(self.points, self.tspans, t)
        pos_d = np.array([0.0,0.0,2.0])
        self.x_d = pos_d[0]
        self.y_d = pos_d[1]
        self.z_d = pos_d[2]

        # PID parameters
        Ki_vh = 0.1
        Kd_vh = 0.0
        MAX_CONTROL_Velocity_x = 3
        MAX_CONTROL_Velocity_y = 3
        MAX_CONTROL_ANGLE_ROLL = np.deg2rad(30)
        MAX_CONTROL_ANGLE_PITCH = np.deg2rad(30)

        Kp_RP = 5
        Kp_RP_rate = 0.5
        Ki_RP_rate = 0.01
        Kd_RP_rate = 0.0  # 0.001
        Kp_Y = 5
        Kp_Y_rate = 1
        Ki_Y_rate = 0.8
        Kd_Y_rate = 0
        I_clamp_att = 0.5

        Kp_th = 4
        Ki_th = 0.15
        Kd_th = 3

        dt_pos = 1/100
        dt_att = 1/500

        # obs = self.env.previous_obs[-1] # (13,)
        pos = obs[0:3]  # should be same as pos
        ori = quat2euler_raw(obs[3:7])  # roll, pitch, yaw
        vel = obs[7:10]  # should be same as qvel
        ang_vel = obs[10:13]  # p, q, r

        dt_pos_ctrl = self.env.timestep * self.env.dt - self.time_ctrlPos_prv
        dt_att_ctrl = self.env.timestep * self.env.dt - self.time_ctrlAtt_prv

        ############# Position Control #############
        if dt_pos_ctrl >= dt_pos:
            self.time_ctrlPos_prv = self.env.timestep * self.env.dt

            self.vx_d = clamp(MAX_CONTROL_Velocity_x, self.Kp_Xh * (self.x_d - pos[0]))
            vx_err = (self.vx_d - vel[0])
            self.vx_I += vx_err * dt_pos
            self.vx_I = clamp(I_clamp_att, self.vx_I)
            pitch_d_raw = (self.Kp_vh * vx_err + Ki_vh * self.vx_I + Kd_vh * (vx_err - self.vx_err_prv) / dt_pos)
            self.pitch_d = clamp(MAX_CONTROL_ANGLE_PITCH, pitch_d_raw)
            self.vx_err_prv = vx_err

            self.vy_d = clamp(MAX_CONTROL_Velocity_y, self.Kp_Xh * (self.y_d - pos[1]))
            vy_err = (self.vy_d - vel[1])
            self.vy_I += vy_err * dt_pos
            self.vy_I = clamp(I_clamp_att, self.vy_I)
            roll_d_raw = -(self.Kp_vh * vy_err + Ki_vh * self.vy_I + Kd_vh * (vy_err - self.vy_err_prv) / dt_pos)
            self.roll_d = clamp(MAX_CONTROL_ANGLE_ROLL, roll_d_raw)
            self.vy_err_prv = vy_err

        # self.pitch_d = 0 # np.deg2rad(10) # np.deg2rad(30)
        # self.roll_d = 0 # np.deg2rad(10)
        ############# Attitude Control #############
        if dt_att_ctrl >= dt_att:
            self.time_ctrlAtt_prv = self.env.timestep * self.env.dt
            # Roll
            self.roll_rate_d = Kp_RP * (self.roll_d - ori[0])
            roll_rate_err = (self.roll_rate_d - ang_vel[0])
            self.roll_I += roll_rate_err * dt_att
            self.roll_I = clamp(I_clamp_att, self.roll_I)
            Roll_raw = Kp_RP_rate * roll_rate_err + Ki_RP_rate * self.roll_I + Kd_RP_rate * (roll_rate_err - self.roll_err_prv) / dt_att
            Roll_command = clamp(1, Roll_raw)
            self.roll_err_prv = roll_rate_err
            # print(Roll_command)
            # Pitch
            self.pitch_rate_d = Kp_RP * (self.pitch_d - ori[1])
            pitch_rate_err = (self.pitch_rate_d - ang_vel[1])
            self.pitch_I += pitch_rate_err * dt_att
            self.pitch_I = clamp(I_clamp_att, self.pitch_I)
            Pitch_raw = Kp_RP_rate * pitch_rate_err + Ki_RP_rate * self.pitch_I + Kd_RP_rate * (pitch_rate_err - self.pitch_err_prv) / dt_att
            Pitch_command = clamp(1, Pitch_raw)
            self.pitch_err_prv = pitch_rate_err
            # print(Pitch_command)
            # Yaw
            self.yaw_rate_d = Kp_Y * (self.yaw_d - ori[2])
            yaw_rate_err = (self.yaw_rate_d - ang_vel[2])
            self.yaw_I += yaw_rate_err * dt_att
            self.yaw_I = clamp(I_clamp_att, self.yaw_I)
            Yaw_raw = Kp_Y_rate * yaw_rate_err + Ki_Y_rate * self.yaw_I + Kd_Y_rate * (yaw_rate_err - self.yaw_err_prv) / dt_att
            Yaw_command = clamp(1, Yaw_raw)
            self.yaw_err_prv = yaw_rate_err
            # print(Yaw_command)

            # Thrust Control
            z_err = (self.z_d - pos[2])
            self.z_I += z_err * dt_att
            self.z_I = clamp(I_clamp_att, self.z_I)
            Thrust_raw = Kp_th * z_err + Ki_th * self.z_I + Kd_th * (z_err - self.z_err_prv) / dt_att + 0.5  # 0.609
            Thrust_command = clamp(1, Thrust_raw)
            self.z_err_prv = z_err

            ControlInput0 = Thrust_command - Pitch_command  # front
            ControlInput1 = Thrust_command + Pitch_command  # back
            ControlInput2 = Thrust_command - Roll_command  # right
            ControlInput3 = Thrust_command + Roll_command  # left

            Thrust0 = ControlInput0
            Thrust1 = ControlInput1
            Thrust2 = ControlInput2
            Thrust3 = ControlInput3
            Thrust4 = Yaw_command
            Thrust5 = Yaw_command

            ctrl = np.array([Thrust0, Thrust1, Thrust2, Thrust3, Thrust4, Thrust5])

            # put the controller here. This function is called inside the simulation.
            # self.data.ctrl[0] = -29.8451302  # set to -29.8451302 => Drive Flapping Kinematics
            # self.data.ctrl[1] = Thrust0  # front
            # self.data.ctrl[2] = Thrust1  # back
            # self.data.ctrl[3] = Thrust2  # left
            # self.data.ctrl[4] = Thrust3  # right
            # self.data.ctrl[5] = Thrust4  # left H
            # self.data.ctrl[6] = Thrust5  # right H
            # print(data.ctrl)
            return ctrl

def clamp(limit, ref):
    if ref < -limit:
        output = -limit
    elif ref > limit:
        output = limit
    else:
        output = ref
    return output


def trajectory_generator(points, tspans, t):
    # Inputs: points - 3xn list = [(0,0,0),...]; tspans - 1xn list
    # Output: pos_d
    if len(points) != len(tspans) + 1:
        print("length not compatible: len(points) not equal len(tspans)+1")

    for i in range(len(tspans)):
        if t <= sum(tspans[:i + 1]):
            # print(sum(tspans[:i+1]))
            pos_d = [points[i, j] + (points[i + 1, j] - points[i, j]) * (t - sum(tspans[:i])) / tspans[i] for j in
                     range(3)]
            return pos_d
    if t > sum(tspans):
        return points[-1]

