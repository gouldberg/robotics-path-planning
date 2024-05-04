"""
Class of n-link arm in 3D
Reference:
 https://github.com/AtsushiSakai/PythonRobotics
 https://github.com/karaage0703/mycobot_pi/tree/main
"""

import numpy as np
import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random

import matplotlib.animation as animation
import glob
from PIL import Image

# for check
from scipy.spatial.transform import Rotation


#######################################################################################################
# -----------------------------------------------------------------------------------------------------
# IMPORTANT !!! : [** DH **]  marked is required to be modified if you change DH param setting
# -----------------------------------------------------------------------------------------------------
#######################################################################################################


# -----------------------------------------------------------------------------------------------------
# helper functions
# -----------------------------------------------------------------------------------------------------

def plot_robot_pose(transmat_list, PLOT_AREA):
    n = len(transmat_list)
    x, y, z = np.array([0.] * (n + 1)), np.array([0.] * (n + 1)), np.array([0.] * (n + 1))
    for i in range(n):
        x[i+1] = transmat_list[i][0,3]
        y[i+1] = transmat_list[i][1,3]
        z[i+1] = transmat_list[i][2,3]
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    # ax = Axes3D(fig)

    ax.plot(x, y, z, "o-", color="#00aa00", ms=4, mew=0.5)
    ax.plot([0], [0], [0], "o")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim(-PLOT_AREA, PLOT_AREA)
    ax.set_ylim(-PLOT_AREA, PLOT_AREA)
    ax.set_zlim(-PLOT_AREA, PLOT_AREA)
    return fig


def random_val(min_val, max_val):
    return min_val + random.random() * (max_val - min_val)


# -----------------------------------------------------------------------------------------------------
# helper functions:  angle, quaternion
# 
# ROS right hand rules:  x forward, y left, z up
# https://www.ros.org/reps/rep-0103.html
# https://en.wikipedia.org/wiki/Right-hand_rule
#
# Rotations in 3-dimensions: Euler angles and rotation matrices:
# https://danceswithcode.net/engineeringnotes/rotations_in_3d/rotations_in_3d_part1.html
#
# Euler ZYZ discussion:
# https://robotics.stackexchange.com/questions/16535/why-euler-angle-is-set-to-be-in-zyz-order
#
# Python scrips for all sequence:
# https://programming-surgeon.com/en/euler-angle-python-en/
#
# Euler ZYZ:
# https://mecharithm.com/learning/lesson/explicit-representations-for-the-orientation-euler-angles-16
# ZYZ Euler Angles Representation for the orientation is popular in robotics
# since most six-degree-of-freedom industrial robots have their fourth axis
# as a rotation about the z-axis of the fourth joint, 
# then the fifth is a rotation about the y-axis of the fifth joint, 
# and the final one is a rotation along the z-axis of the end-effector as depicted in the figure below:
#
# https://www.sky-engin.jp/blog/eulerian-angles/
# -----------------------------------------------------------------------------------------------------

# math.atan2(y,x) returns from - math.pi / 2 (radian) to + math.pi / 2 (radian)

def rot2quat(transmat):
    flag = 1
    if transmat[0, 3] < 0:
        flag = -1
    r11 = transmat[0,0]
    r12 = transmat[0,1]
    r13 = transmat[0,2]
    r21 = transmat[1,0]
    r22 = transmat[1,1]
    r23 = transmat[1,2]
    r31 = transmat[2,0]
    r32 = transmat[2,1]
    r33 = transmat[2,2]
    qw = 0.5 * math.sqrt(1 + r11 + r22 + r33)
    qx = flag * (r32 - r23) / (4 * qw)
    qy = flag * (r13 - r31) / (4 * qw)
    qz = flag * (r21 - r12) / (4 * qw)
    return qw, qx, qy, qz


def quat2rpy(quat):
    qw = quat[0]
    qx = quat[1]
    qy = quat[2]
    qz = quat[3]
    alpha = math.atan2(2 * (qy * qz + qw * qx), 1 - 2 * (qx ** 2 + qy ** 2))
    beta = math.atan2(-2 * (qx * qz - qw * qy), math.sqrt(4 * (qy * qz + qw * qx) ** 2 + (1 - 2 * (qx ** 2 + qy ** 2)) ** 2))
    gamma = math.atan2(2 * (qx * qy + qw * qz), 1 - 2 * (qy ** 2 + qz ** 2))
    return alpha, beta, gamma


def rot2euler(transmat):
    flag = 1
    if transmat[0, 3] < 0:
        flag = -1
    r11 = transmat[0,0]
    r12 = transmat[0,1]
    r13 = transmat[0,2]
    r21 = transmat[1,0]
    r22 = transmat[1,1]
    r23 = transmat[1,2]
    r31 = transmat[2,0]
    r32 = transmat[2,1]
    r33 = transmat[2,2]
    alpha = math.atan2(flag * math.sqrt(r13 ** 2 + r23 ** 2), r33)
    beta = math.atan2(flag * r23, flag * r13)
    gamma = math.atan2(flag * r32, -1 * flag * r31)
    return alpha, beta, gamma

# ----------
# https://programming-surgeon.com/en/euler-angle-python-en/
# rot2euler2 is .. ??
def rot2euler2(transmat):
    alpha = math.atan2(transmat[1][2], transmat[0][2])
    if not (-math.pi / 2 <= alpha <= math.pi / 2):
        alpha = math.atan2(transmat[1][2], transmat[0][2]) + math.pi
    if not (-math.pi / 2 <= alpha <= math.pi / 2):
        alpha = math.atan2(transmat[1][2], transmat[0][2]) - math.pi
    beta = math.atan2(
        transmat[0][2] * math.cos(alpha) + transmat[1][2] * math.sin(alpha),
        transmat[2][2])
    gamma = math.atan2(
        -transmat[0][0] * math.sin(alpha) + transmat[1][0] * math.cos(alpha),
        -transmat[0][1] * math.sin(alpha) + transmat[1][1] * math.cos(alpha))

    return alpha, beta, gamma


# -----------------------------------------------------------------------------------------------------
# forward kinematics
# -----------------------------------------------------------------------------------------------------

def transformation_matrix(link_params):

    # a:        length of the common normal, Assuming a revolute joint, this is the radius about previous z
    # alpha:    angle about common normal, from old z axis to new z axis
    # d:        offset along previous z to the common normal
    # theta:    angle about previous z from old x to new x

    #### [** DH **] ####
    # ----------
    # [a, alpha, d, theta]
    # a = link_params[0]
    # alpha = link_params[1]
    # d = link_params[2]
    # theta = link_params[3]

    # ----------
    # [theta, alpha, a, d]
    a = link_params[2]
    alpha = link_params[1]
    d = link_params[3]
    theta = link_params[0]
    # ----------
    ###################

    st = math.sin(theta)
    ct = math.cos(theta)
    sa = math.sin(alpha)
    ca = math.cos(alpha)

    #### [** DH **] ####
    # THIS IS NOT TO BE APPLIED: 'modified (proximal) DH parameters'
    # Coordinate of O(i) is put on the axis i
    # transformation matrix is given by : Rot(a(n-1)) * Trans(alpha(n-1)) * Rot(theta(n)) * Trans(d(n))
    # transmat = np.array([[ct, -st, 0, a],
    #                     [ca * st, ca * ct, -sa, -sa * d],
    #                     [sa * st, sa * ct, ca, ca * d],
    #                     [0, 0, 0, 1]])

    # HERE IS REQUIRED based on: 'classic (distal) DH parameters'
    # https://en.wikipedia.org/wiki/Denavit%E2%80%93Hartenberg_parameters
    # Coordinate of O(i) is put on the axis i + 1
    # transformation matrix is given by : Trans(d(n)) * Rot(theta(n)) * Trans(d(n)) * Rot(alpha(n))
    transmat = np.array([[ct, -st * ca, st * sa, a * ct],
                        [st, ct * ca, -ct * sa, a * st],
                        [0, sa, ca, d],
                        [0, 0, 0, 1]])
    ###################
    
    return transmat


def basic_jacobian(trans_prev, ee_pos):
    pos_prev = np.array(trans_prev[0:3, 3])
    # ---------
    # z_axis: joint rotate around z-axis (in definition)
    z_axis_prev = np.array(trans_prev[0:3, 2])
    # ---------
    # This is NOT BASIC JACOBIAN, just Jacobian (wakariyasui robot system nyuumon:  p.119)
    basic_jacob = np.hstack((np.cross(z_axis_prev, ee_pos - pos_prev), z_axis_prev))
    return basic_jacob


def forward_kinematics(link_list):
    n_link = len(link_list)
    transmat_local_list, transmat_global_list, basic_jacobian_mat_list = [], [], []
    transmat_global = np.identity(4)

    for i in range(n_link):
        transmat_local = transformation_matrix(link_list[i])
        transmat_global = np.dot(transmat_global, transmat_local)
        transmat_local_list.append(transmat_local)
        transmat_global_list.append(transmat_global)
        # print(f'{i} : trans matrix global:')
        # print(f'{transmat_global}')

    # ----------
    # jacobian
    trans = np.identity(4)
    ee_pos = transmat_global[0:3, 3]
    for i in range(n_link):
        basic_jacobian_mat_list.append(basic_jacobian(trans, ee_pos))
        trans = np.dot(trans, transformation_matrix(link_list[i]))

    # basic jacobian is not BASIC, it is just Jacobian
    return transmat_local_list, transmat_global_list, np.array(basic_jacobian_mat_list).T


#######################################################################################################
# -----------------------------------------------------------------------------------------------------
# Set DH parameters and plot Robot pose in 3D
# -----------------------------------------------------------------------------------------------------

# Set DH parameters (Denavit-Hartenberg)
# https://en.wikipedia.org/wiki/Denavit%E2%80%93Hartenberg_parameters

# [a, alpha, d, theta]
# a:        length of the common normal, Assuming a revolute joint, this is the radius about previous z
# alpha:    angle about common normal, from old z axis to new z axis
# d:        offset along previous z to the common normal
# theta:    angle about previous z from old x to new x

link_list = [
    [0.,        math.pi / 2.,   0.13156,    0.],
    [-0.1104,   0.,             0.,         0.],
    [-0.096,    0.,             0.,         0.],
    [0.,        math.pi / 2.,   0.06639,    0.],
    [0.,        math.pi / 2.,   0.07318,    0.],
    [0.,        0.,             0.0436,     0.]
]

link_list = [
    [0.,        0.,             0.05,       0.],
    [0.05,      math.pi / 2.,   0.,         0.],
    [0.,        0.,             0.05,       0.],
    [0.,        0.,             0.05,       0.],
    [0.,        0.,             0.05,       0.],
    [0.,        0.,             0.05,       0.]
]

link_list = [
    [0.,        0.,             0.05,       0.],
    [0.05,      math.pi / 2.,   0.,         0.],
    [0.,        0.,             0.05,       math.pi / 2.],
    [0.05,      0.,             0.,         0.],
    [0.05,      0.,             0.,         0.],
    [0.05,      0.,             0.,         0.]
]

# check transformation matrix based on dh_params_list
for i in range(len(link_list)):
    transmat_local = transformation_matrix(link_list[i])
    print(f'{i} : trans matrix local :')
    print(f'{transmat_local}')


# ----------
# transformation (forward kinematics)
transmat_local_list, transmat_global_list, basic_jacobian_mat_list = forward_kinematics(link_list)

# ee (end effector) position
print(transmat_global_list[-1][0:3,3])

# plot
fig = plot_robot_pose(transmat_list=transmat_global_list, PLOT_AREA=0.4)
fig.show()



#######################################################################################################
# -----------------------------------------------------------------------------------------------------
# Forward Kinematics:  reset angle (manually or random)
# -----------------------------------------------------------------------------------------------------

# reset angle (radian) by each joint
# joint_angles = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

joint_angles = [random_val(-1, 1) for _ in range(len(link_list))]
print(f'joint angles : {joint_angles}')

for i in range(len(link_list)):
    link_list[i][3] = joint_angles[i]


# check transformation matrix (local) updated
for i in range(len(link_list)):
    transmat_local = transformation_matrix(link_list[i])
    print(f'{i} : trans matrix local:')
    print(f'{transmat_local}')


# ----------
# transformation (forward kinematics)
transmat_local_list, transmat_global_list, basic_jacobian_mat_list = forward_kinematics(link_list)

# this is not recommended
np.round(rot2euler(transmat=transmat_global_list[-1]), 3)

# Here original implementation !!!
# alpha(1st angle) is from -math.pi / 2 to + math.pi / 2
# this is ZYZ euler AND alpha(1st angle) is from -math.pi / 2 to + math.pi / 2
np.round(rot2euler2(transmat=transmat_global_list[-1]), 3)

# Scipy Rotation:  1st and 3rd angle is from -math.pi / 2 to + math.pi / 2 for 'xyz', 'zxz'
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.as_euler.html
np.round(Rotation.from_matrix(transmat_global_list[-1][0:3,0:3]).as_euler('xyz', degrees=False), 3)
np.round(Rotation.from_matrix(transmat_global_list[-1][0:3,0:3]).as_euler('zyz', degrees=False), 3)

# for reference:
# roll, pitch, yaw (zyx)
np.round(quat2rpy(rot2quat(transmat_global_list[-1])), 3)
np.round(Rotation.from_matrix(transmat_global_list[-1][0:3,0:3]).as_euler('zxy', degrees=False), 3)


# ----------
# plot
fig = plot_robot_pose(transmat_list=transmat_global_list, PLOT_AREA=0.2)
fig.show()



#######################################################################################################
# -----------------------------------------------------------------------------------------------------
# Inverse Kinematics
#  - Newton Raphson vs. Levenberg–Marquardt etc...
# -----------------------------------------------------------------------------------------------------

def inverse_kinematics(iter_cnt, link_list, ref_ee_pose, plot=False, plot_anim=False, save_path='./04_output/robot_inverse_kinematics.gif'):
    
    if plot:
        PLOT_AREA = 0.4
        plt.close()

    if plot and plot_anim:
        ims = []
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        # ax = Axes3D(fig)
        ax.plot([0], [0], [0], "x")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_xlim(-PLOT_AREA, PLOT_AREA)
        ax.set_ylim(-PLOT_AREA, PLOT_AREA)
        ax.set_zlim(-PLOT_AREA, PLOT_AREA)
        ax.plot([ref_ee_pose[0]], [ref_ee_pose[1]], [ref_ee_pose[2]], "o")

    for cnt in range(iter_cnt):
        print(f'### -- cnt : {cnt + 1} / {iter_cnt}')
        _, transmat_global_list, basic_jacobian_mat = forward_kinematics(link_list)
        x, y, z = transmat_global_list[-1][0:3, 3]
        # ----------
        # Both are OK...
        # alpha, beta, gamma = rot2euler(transmat_global_list[-1])
        alpha, beta, gamma = rot2euler2(transmat_global_list[-1])
        # ----------
        ee_pose = [x, y, z, alpha, beta, gamma]
        diff_pose = np.array(ref_ee_pose) - np.array(ee_pose)

        # ----------
        # K_zyz is ZYZ rotation angular velocity vector's pose (jissen-robot seigyo p. 82 is ZYX)
        K_zyz = np.array([
            [0, -math.sin(alpha), math.cos(alpha) * math.sin(beta)],
            [0, math.cos(alpha), math.sin(alpha) * math.sin(beta)],
            [1, 0, math.cos(beta)]])
        # ----------
        # K_alpha in p.83 (6.42)
        K_alpha = np.identity(6)
        K_alpha[3:, 3:] = K_zyz
        
        # -----------
        # Newton Raphson
        # numpy.linalg.pinv:  calculate the generalized inverse of a matrix using its singular-value decomposition (SVD)
        # and including all large singular values.
        # This is based on Newton-Raphson (jissen robot seigyo p.100, (7.33) or wakariyasui robot system ,, p.131 (8.17))
        # ----------
        diff_joint_angle_list = np.dot(
            np.dot(np.linalg.pinv(basic_jacobian_mat), K_alpha),
            np.array(diff_pose))

        # ----------
        # Levenberg–Marquardt:  this achieves smooth operation
        # ----------
        # weight_mat = np.diag([1., 1., 1., math.pi/2., math.pi/2., math.pi/2.])
        # identity_mat = np.identity(6)
        # jv = np.dot(K_alpha, basic_jacobian_mat)
        # eps = 0.00001
        # tmp = np.linalg.pinv(np.dot(np.dot(jv.T, weight_mat), jv) + eps * identity_mat)
        # diff_joint_angle_list = np.dot(np.dot(np.dot(tmp, jv.T), weight_mat), np.array(diff_pose))

        # ----------
        #### [** DH **] ####
        # theta_index_in_link_list = 3
        theta_index_in_link_list = 0
        ####################
        diff_denom = 200.
        # ----------
        for i in range(len(link_list)):
            link_list[i][theta_index_in_link_list] += diff_joint_angle_list[i] / diff_denom

        if plot:
            n = len(transmat_global_list)
            x, y, z = np.array([0.] * (n + 1)), np.array([0.] * (n + 1)), np.array([0.] * (n + 1))
            for i in range(n):
                x[i+1] = transmat_global_list[i][0,3]
                y[i+1] = transmat_global_list[i][1,3]
                z[i+1] = transmat_global_list[i][2,3]

            if plot_anim:
                im = ax.plot(x, y, z, "o-", color="#00aa00", ms=4, mew=0.5)
                ims.append(im)

            elif cnt % 10 == 0:
                fig = plt.figure(figsize=(8, 8))
                ax = fig.add_subplot(111, projection='3d')
                # ax = Axes3D(fig)
                ax.plot([0], [0], [0], "x")
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_zlabel("z")
                ax.set_xlim(-PLOT_AREA, PLOT_AREA)
                ax.set_ylim(-PLOT_AREA, PLOT_AREA)
                ax.set_zlim(-PLOT_AREA, PLOT_AREA)
                ax.plot([ref_ee_pose[0]], [ref_ee_pose[1]], [ref_ee_pose[2]], "o")
                ax.plot(x, y, z, "o-", color="#00aa00", ms=4, mew=0.5)
                plt.savefig(f'./04_output/pngs/sample_{str(cnt).zfill(3)}.png')

                if cnt == 0:
                    fig.show()  

    if plot and plot_anim:
        print('animation creating !!')
        ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat=False)
        ani.save(save_path, writer="pillow")
        plt.close()
        print(f'animation saved DONE !! : {save_path}')

# Set DH parameters (Denavit-Hartenberg)

# [a, alpha, d, theta]
# a: translation to x-axis
# alpha: rotation around x (clock-wise)
# d: translation to z-axis
# theta: rotation around z (clock-wise)

# all theta (4th column) are zero
link_list = [
    [0.,        math.pi / 2.,   0.13156,    0.],
    [-0.1104,   0.,             0.,         0.],
    [-0.096,    0.,             0.,         0.],
    [0.,        math.pi / 2.,   0.06639,    0.],
    [0.,        math.pi / 2.,   0.07318,    0.],
    [0.,        0.,             0.0436,     0.]
]

# [theta, alpha, a, d]
link_list = [
    [0.,    math.pi / 2.,    0.,         0.13156],
    [0.,    0.,             -0.1104,    0.],
    [0.,    0.,             -0.096,     0.],
    [0.,    math.pi / 2.,    0.,         0.06639],
    [0.,    -math.pi / 2.,   0.,         0.07318],
    [0.,    0.,             0.,         0.0436]
]

# link_list = [
#     [0., 0., 0.05, 0.],
#     [0.05, math.pi / 2., 0., 0.],
#     [0., 0., 0.05, 0.],
#     [0., 0., 0.05, 0.],
#     [0., 0., 0.05, 0.],
#     [0., 0., 0.05, 0.]
# ]

# link_list = [
#     [0., 0., 0.05, 0.],
#     [0.05, math.pi / 2., 0., 0.],
#     [0., 0., 0.05, math.pi / 2.],
#     [0.05, 0., 0., 0.],
#     [0.05, 0., 0., 0.],
#     [0.05, 0., 0., 0.]
# ]


plt.close()

save_path = './04_output/robot_inverse_kinematics.gif'
iter_cnt = 500
ref_ee_pose = [0.15, 0.2, 0.1, 0, 0, 0.2]
# ref_ee_pose = [0.15, 0.1, 0.1, 0, 0, 0.2]

plot=True
plot_anim=True

inverse_kinematics(iter_cnt, link_list, ref_ee_pose=ref_ee_pose, plot=True, plot_anim=plot_anim, save_path=save_path)



#######################################################################################################
# -----------------------------------------------------------------------------------------------------
# check
# -----------------------------------------------------------------------------------------------------

link_list = [
    [0.,    math.pi / 2.,    0.,         0.13156],
    [0.,    0.,             -0.1104,    0.],
    [0.,    0.,             -0.096,     0.],
    [0.,    math.pi / 2.,    0.,         0.06639],
    [0.,    -math.pi / 2.,   0.,         0.07318],
    [0.,    0.,             0.,         0.0436]
]

ref_ee_pose = [0.15, 0.2, 0.1, 0, 0, 0.2]


# ----------
_, transmat_global_list, basic_jacobian_mat = forward_kinematics(link_list)
x, y, z = transmat_global_list[-1][0:3, 3]


alpha, beta, gamma = rot2euler2(transmat_global_list[-1])
ee_pose = [x, y, z, alpha, beta, gamma]

# this is size 6 * 1
diff_pose = np.array(ref_ee_pose) - np.array(ee_pose)

K_zyz = np.array([
    [0, -math.sin(alpha), math.cos(alpha) * math.sin(beta)],
    [0, math.cos(alpha), math.sin(alpha) * math.sin(beta)],
    [1, 0, math.cos(beta)]])
K_alpha = np.identity(6)
K_alpha[3:, 3:] = K_zyz

diff_joint_angle_list = np.dot(
    np.dot(np.linalg.pinv(basic_jacobian_mat), K_alpha),
    np.array(diff_pose))

# this is size 6 * 1
print(diff_joint_angle_list)


###########################
weight_mat = np.diag([1., 1., 1., math.pi/2., math.pi/2., math.pi/2.])
identity_mat = np.identity(6)
jv = np.dot(K_alpha, basic_jacobian_mat)
eps = 0.001
tmp = np.linalg.pinv(np.dot(np.dot(jv.T, weight_mat), jv) + eps * identity_mat)
diff_joint_angle_list = np.dot(np.dot(np.dot(tmp, jv.T), weight_mat), np.array(diff_pose))
