import numpy as np
from func_initial_joint_angles import *

def FDC_ctrlInput2Length(FDC_ctrl, FDC_range):  # on one side
    for i in range(len(FDC_ctrl)):
        FDC_ctrl[i] = min(max(FDC_ctrl[i],0),1)  # limit control input to [0,1]
    # Convert control input to FDC lengths
    L3a = FDC_range[0, 0] + FDC_ctrl[0] * (FDC_range[0, 1] - FDC_range[0, 0])
    L5a = FDC_range[1, 0] + FDC_ctrl[1] * (FDC_range[1, 1] - FDC_range[1, 0])
    L5b = FDC_range[2, 0] + FDC_ctrl[2] * (FDC_range[2, 1] - FDC_range[2, 0])
    FDC_length = [L3a, L5a, L5b]
    return FDC_length


def FDC_length2JointAngle(FDC_length, theta1):  # on one side
    # FDC length: L3a, L5a, L5b
    # Calculate Joint angles
    Thetas = FDC_theta(theta1, FDC_length[0], FDC_length[1], FDC_length[2]).reshape(7, )
    # Apply wing joint angles
    J5_ = Thetas[3]
    J6_ = Thetas[4]
    J5_d = J5_ - np.deg2rad(11.345825599281223)  # convert to Mujoco Reference
    J6_d = J6_ + np.deg2rad(27.45260202)  # -J5_
    return J5_d, J6_d


def clampFDC(FDC_length, FDC_range):
    for i in range(2):
        for j in range(len(FDC_length[0,:])):
            FDC_length[i,j] = min(max(FDC_length[i,j],FDC_range[j,0]),FDC_range[j,1])
    # print(FDC_length)
    return FDC_length

def rk4(dynamics_func, state, reference, dt, *args):
    k1 = dynamics_func(state, reference, *args)
    k2 = dynamics_func(state + 0.5 * dt * k1, reference, *args)
    k3 = dynamics_func(state + 0.5 * dt * k2, reference, *args)
    k4 = dynamics_func(state + dt * k3, reference, *args)
    return state + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

def theta1_dynamics(state, reference, kd_theta1):  # Motor dynmics
    theta1, d_theta1 = state
    d_theta1_ref = reference
    dd_theta1 = kd_theta1 * (d_theta1_ref - d_theta1)
    return np.array([d_theta1, dd_theta1])

def fdc_dynamics(state, reference, kp_fdc, kd_fdc):
    FDC_pos, FDC_vel = state
    FDC_ref = reference
    FDC_acc = kp_fdc * (FDC_ref - FDC_pos) + kd_fdc * (0 - FDC_vel)
    return np.array([FDC_vel, FDC_acc])