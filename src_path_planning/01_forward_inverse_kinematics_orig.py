"""
Class of n-link arm in 3D
Author: Takayuki Murooka (takayuki5168)
Reference: https://github.com/AtsushiSakai/PythonRobotics
"""

import numpy as np
import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random

import matplotlib.animation as animation
import glob
from PIL import Image

PLOT_AREA = 0.5


#######################################################################################################
# -----------------------------------------------------------------------------------------------------
# class Link
# -----------------------------------------------------------------------------------------------------

class Link:
    def __init__(self, dh_params):
        self.dh_params_ = dh_params

    def transformation_matrix(self):
        # theta = self.joint_angle_
        theta = self.dh_params_[0]
        alpha = self.dh_params_[1]
        a = self.dh_params_[2]
        d = self.dh_params_[3]

        st = math.sin(theta)
        ct = math.cos(theta)
        sa = math.sin(alpha)
        ca = math.cos(alpha)

        # trans = np.array([[ct, -st, 0, a],
        #                     [ca * st, ca * ct, -sa, -sa * d],
        #                     [sa * st, sa * ct, ca, ca * d],
        #                     [0, 0, 0, 1]])

        trans = np.array([[ct, -st * ca, st * sa, a * ct],
                          [st, ct * ca, -ct * sa, a * st],
                          [0, sa, ca, d],
                          [0, 0, 0, 1]])

        return trans

    @staticmethod
    def basic_jacobian(trans_prev, ee_pos):
        pos_prev = np.array(
            [trans_prev[0, 3], trans_prev[1, 3], trans_prev[2, 3]])
        z_axis_prev = np.array(
            [trans_prev[0, 2], trans_prev[1, 2], trans_prev[2, 2]])

        basic_jacobian = np.hstack(
            (np.cross(z_axis_prev, ee_pos - pos_prev), z_axis_prev))
        return basic_jacobian


#######################################################################################################
# -----------------------------------------------------------------------------------------------------
# class NLinkArm
# -----------------------------------------------------------------------------------------------------

class NLinkArm:
    def __init__(self, dh_params_list):
        self.link_list = []
        for i in range(len(dh_params_list)):
            self.link_list.append(Link(dh_params_list[i]))

    @staticmethod
    def convert_joint_angles_sim_to_mycobot(joint_angles):
        """convert joint angles simulator to mycobot

        Args:
            joint_angles ([float]): [joint angles(radian)]

        Returns:
            [float]: [joint angles calculated(radian)]
        """
        conv_mul = [-1.0, -1.0, 1.0, -1.0, -1.0, -1.0]
        conv_add = [0.0, -math.pi / 2, 0.0, -math.pi / 2, math.pi / 2, 0.0]

        joint_angles = [joint_angles * conv_mul for (joint_angles, conv_mul) in zip(joint_angles, conv_mul)]
        joint_angles = [joint_angles + conv_add for (joint_angles, conv_add) in zip(joint_angles, conv_add)]

        joint_angles_lim = []
        for joint_angle in joint_angles:
            while joint_angle > math.pi:
                joint_angle -= 2 * math.pi

            while joint_angle < -math.pi:
                joint_angle += 2 * math.pi

            joint_angles_lim.append(joint_angle)

        return joint_angles_lim

    def transformation_matrix(self):
        trans = np.identity(4)
        for i in range(len(self.link_list)):
            trans = np.dot(trans, self.link_list[i].transformation_matrix())
        return trans

    def forward_kinematics(self, plot=False):
        trans = self.transformation_matrix()

        x = trans[0, 3]
        y = trans[1, 3]
        z = trans[2, 3]
        alpha, beta, gamma = self.euler_angle()

        if plot:
            self.fig = plt.figure()
            # self.ax = Axes3D(self.fig)
            self.ax = self.fig.add_subplot(111, projection='3d')

            x_list = []
            y_list = []
            z_list = []

            trans = np.identity(4)

            x_list.append(trans[0, 3])
            y_list.append(trans[1, 3])
            z_list.append(trans[2, 3])
            for i in range(len(self.link_list)):
                trans = np.dot(trans, self.link_list[i].transformation_matrix())
                x_list.append(trans[0, 3])
                y_list.append(trans[1, 3])
                z_list.append(trans[2, 3])

            self.ax.plot(x_list, y_list, z_list, "o-", color="#00aa00", ms=4,
                         mew=0.5)
            self.ax.plot([0], [0], [0], "o")

            self.ax.set_xlim(-PLOT_AREA, PLOT_AREA)
            self.ax.set_ylim(-PLOT_AREA, PLOT_AREA)
            self.ax.set_zlim(-PLOT_AREA, PLOT_AREA)

            plt.show()

        return [x, y, z, alpha, beta, gamma]

    def basic_jacobian(self):
        ee_pos = self.forward_kinematics()[0:3]
        basic_jacobian_mat = []

        trans = np.identity(4)
        for i in range(len(self.link_list)):
            basic_jacobian_mat.append(
                self.link_list[i].basic_jacobian(trans, ee_pos))
            trans = np.dot(trans, self.link_list[i].transformation_matrix())

        return np.array(basic_jacobian_mat).T

    def inverse_kinematics(self, iter_cnt, ref_ee_pose, plot=False, plot_anim=False, save_path='./04_output/robot_inverse_kinematics.gif'):
        if plot:
            PLOT_AREA = 0.4

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
            print(f'cnt : {cnt + 1} / {iter_cnt}')
            ee_pose = self.forward_kinematics()
            diff_pose = np.array(ref_ee_pose) - ee_pose

            basic_jacobian_mat = self.basic_jacobian()
            alpha, beta, gamma = self.euler_angle()

            K_zyz = np.array(
                [[0, -math.sin(alpha), math.cos(alpha) * math.sin(beta)],
                 [0, math.cos(alpha), math.sin(alpha) * math.sin(beta)],
                 [1, 0, math.cos(beta)]])
            K_alpha = np.identity(6)
            K_alpha[3:, 3:] = K_zyz

            theta_dot = np.dot(
                np.dot(np.linalg.pinv(basic_jacobian_mat), K_alpha),
                np.array(diff_pose))

            # -----------
            self.update_joint_angles(theta_dot / 100.)
            # self.update_joint_angles(theta_dot / 200.)
            # -----------

            if plot:
                x_list = []
                y_list = []
                z_list = []

                trans = np.identity(4)

                x_list.append(trans[0, 3])
                y_list.append(trans[1, 3])
                z_list.append(trans[2, 3])
                for i in range(len(self.link_list)):
                    trans = np.dot(trans, self.link_list[i].transformation_matrix())
                    x_list.append(trans[0, 3])
                    y_list.append(trans[1, 3])
                    z_list.append(trans[2, 3])

                # ----------
                if plot_anim:
                    im = ax.plot(x_list, y_list, z_list, "o-", color="#00aa00", ms=4, mew=0.5)
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
                    ax.plot(x_list, y_list, z_list, "o-", color="#00aa00", ms=4, mew=0.5)
                    plt.savefig(f'./04_output/pngs/sample_{str(cnt).zfill(3)}.png')

                    if cnt == 0:
                        fig.show()  

        if plot and plot_anim:
            ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat=False)
            ani.save(save_path, writer="pillow")
            plt.close()
            print(f'animation saved DONE !! : {save_path}')

    def euler_angle(self):
        trans = self.transformation_matrix()

        alpha = math.atan2(trans[1][2], trans[0][2])
        if not (-math.pi / 2 <= alpha <= math.pi / 2):
            alpha = math.atan2(trans[1][2], trans[0][2]) + math.pi
        if not (-math.pi / 2 <= alpha <= math.pi / 2):
            alpha = math.atan2(trans[1][2], trans[0][2]) - math.pi
        beta = math.atan2(
            trans[0][2] * math.cos(alpha) + trans[1][2] * math.sin(alpha),
            trans[2][2])
        gamma = math.atan2(
            -trans[0][0] * math.sin(alpha) + trans[1][0] * math.cos(alpha),
            -trans[0][1] * math.sin(alpha) + trans[1][1] * math.cos(alpha))

        return alpha, beta, gamma

    def send_angles(self, joint_angle_list):
        for i in range(len(self.link_list)):
            self.link_list[i].dh_params_[0] = joint_angle_list[i]

    def update_joint_angles(self, diff_joint_angle_list):
        for i in range(len(self.link_list)):
            self.link_list[i].dh_params_[0] += diff_joint_angle_list[i]

    def get_angles(self):
        joint_angles = []
        for i in range(len(self.link_list)):
            joint_angle = self.link_list[i].dh_params_[0]
            while joint_angle > math.pi:
                joint_angle -= 2 * math.pi

            while joint_angle < -math.pi:
                joint_angle += 2 * math.pi

            joint_angles.append(joint_angle)

        return joint_angles

    def plot(self):
        self.fig = plt.figure()
        # self.ax = Axes3D(self.fig)
        self.ax = self.fig.add_subplot(111, projection='3d')

        x_list = []
        y_list = []
        z_list = []

        trans = np.identity(4)

        x_list.append(trans[0, 3])
        y_list.append(trans[1, 3])
        z_list.append(trans[2, 3])
        for i in range(len(self.link_list)):
            trans = np.dot(trans, self.link_list[i].transformation_matrix())
            x_list.append(trans[0, 3])
            y_list.append(trans[1, 3])
            z_list.append(trans[2, 3])

        self.ax.plot(x_list, y_list, z_list, "o-", color="#00aa00", ms=4,
                     mew=0.5)
        self.ax.plot([0], [0], [0], "o")

        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")

        self.ax.set_xlim(-PLOT_AREA, PLOT_AREA)
        self.ax.set_ylim(-PLOT_AREA, PLOT_AREA)
        self.ax.set_zlim(-PLOT_AREA, PLOT_AREA)
        plt.show()


def random_val(min_val, max_val):
    return min_val + random.random() * (max_val - min_val)


#######################################################################################################
# -----------------------------------------------------------------------------------------------------
# Inverse Kinematics
# -----------------------------------------------------------------------------------------------------

# Set DH parameters (Denavit-Hartenberg)

# [theta, alpha, a, d]
# theta: rotation around z (clock-wise)
# alpha: rotation around x (clock-wise)
# a: translation to x-axis
# d: translation to z-axis

# all of theta (first column) are zero.
link_list = [
    [0.,    math.pi / 2,    0.,         0.13156],
    [0.,    0.,             -0.1104,    0.],
    [0.,    0.,             -0.096,     0.],
    [0.,    math.pi / 2,    0.,         0.06639],
    [0.,    -math.pi / 2,   0.,         0.07318],
    [0.,    0.,             0.,         0.0436]
]

# randomly reset theta
joint_angles = [random_val(-1, 1) for _ in range(len(link_list))]
print(f'joint angles : {joint_angles}')
for i in range(len(link_list)):
    link_list[i][0] = joint_angles[i]


mycobot_sim = NLinkArm(link_list)

save_path = './04_output/robot_inverse_kinematics.gif'

iter_cnt = 500
ref_ee_pose = [0.15, 0.2, 0.1, 0, 0, 0.2]

plot=True
plot_anim=True
mycobot_sim.inverse_kinematics(iter_cnt=iter_cnt, ref_ee_pose=ref_ee_pose, plot=plot, plot_anim=plot_anim, save_path=save_path)


