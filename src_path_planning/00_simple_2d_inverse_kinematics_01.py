# Inverse kinematics for an n-link arm using the Jacobian inverse method

import numpy as np
import matplotlib.pyplot as plt

import matplotlib.animation as animation


#######################################################################################################
# -----------------------------------------------------------------------------------------------------
# angle_mod
# -----------------------------------------------------------------------------------------------------

def angle_mod(x, zero_2_2pi=False, degree=False):
    if isinstance(x, float):
        is_float = True
    else:
        is_float = False
    x = np.asarray(x).flatten()
    if degree:
        x = np.deg2rad(x)
    if zero_2_2pi:
        mod_angle = x % (2 * np.pi)
    else:
        mod_angle = (x + np.pi) % (2 * np.pi) - np.pi
    if degree:
        mod_angle = np.rad2deg(mod_angle)

    if is_float:
        return mod_angle.item()
    else:
        return mod_angle


#######################################################################################################
# -----------------------------------------------------------------------------------------------------
# NLinkArm
# -----------------------------------------------------------------------------------------------------

class NLinkArm(object):
    def __init__(self, link_lengths, joint_angles, goal):
        self.n_links = len(link_lengths)
        if self.n_links != len(joint_angles):
            raise ValueError()

        self.link_lengths = np.array(link_lengths)
        self.joint_angles = np.array(joint_angles)
        self.points = [[0, 0] for _ in range(self.n_links + 1)]

        self.lim = sum(link_lengths)
        self.goal = np.array(goal).T
        self.update_points()

    def update_joints(self, joint_angles):
        self.joint_angles = joint_angles

        self.update_points()

    def update_points(self):
        for i in range(1, self.n_links + 1):
            self.points[i][0] = self.points[i - 1][0] + \
                self.link_lengths[i - 1] * \
                np.cos(np.sum(self.joint_angles[:i]))
            self.points[i][1] = self.points[i - 1][1] + \
                self.link_lengths[i - 1] * \
                np.sin(np.sum(self.joint_angles[:i]))

        self.end_effector = np.array(self.points[self.n_links]).T


def inverse_kinematics(link_lengths, joint_angles, goal_pos):
    """
    Calculates the inverse kinematics using the Jacobian inverse method.
    """
    for iteration in range(N_ITERATIONS):
        current_pos = forward_kinematics(link_lengths, joint_angles)
        errors, distance = distance_to_goal(current_pos, goal_pos)
        if distance < 0.1:
            print("Solution found in %d iterations." % iteration)
            return joint_angles, True
        J = jacobian_inverse(link_lengths, joint_angles)
        # ----------
        # new joint angles
        joint_angles = joint_angles + np.matmul(J, errors)
    return joint_angles, False


def forward_kinematics(link_lengths, joint_angles):
    x = y = 0
    for i in range(1, N_LINKS + 1):
        x += link_lengths[i - 1] * np.cos(np.sum(joint_angles[:i]))
        y += link_lengths[i - 1] * np.sin(np.sum(joint_angles[:i]))
    return np.array([x, y]).T


def jacobian_inverse(link_lengths, joint_angles):
    J = np.zeros((2, N_LINKS))
    for i in range(N_LINKS):
        J[0, i] = 0
        J[1, i] = 0
        for j in range(i, N_LINKS):
            J[0, i] -= link_lengths[j] * np.sin(np.sum(joint_angles[:j]))
            J[1, i] += link_lengths[j] * np.cos(np.sum(joint_angles[:j]))

    return np.linalg.pinv(J)


def distance_to_goal(current_pos, goal_pos):
    x_diff = goal_pos[0] - current_pos[0]
    y_diff = goal_pos[1] - current_pos[1]
    return np.array([x_diff, y_diff]).T, np.hypot(x_diff, y_diff)


def ang_diff(theta1, theta2):
    """
    Returns the difference between two angles in the range -pi to +pi
    """
    return angle_mod(theta1 - theta2)


#######################################################################################################
# -----------------------------------------------------------------------------------------------------
# simple inverse kinematics 2D
# -----------------------------------------------------------------------------------------------------

def plot_point_control(arm, save_path):
    fig = plt.figure()
    for i in range(arm.n_links + 1):
        if i is not arm.n_links:
            plt.plot([arm.points[i][0], arm.points[i + 1][0]], [arm.points[i][1], arm.points[i + 1][1]], 'r-')
        plt.plot(arm.points[i][0], arm.points[i][1], 'ko')
    plt.plot(arm.goal[0], arm.goal[1], 'gx')
    plt.plot([arm.end_effector[0], arm.goal[0]], [arm.end_effector[1], arm.goal[1]], 'g--')
    plt.xlim([-arm.lim - 2, arm.lim + 2])
    plt.ylim([-arm.lim - 2, arm.lim + 2])
    plt.savefig(save_path)
    
def get_random_goal():
    from random import random
    SAREA = 15.0
    return [SAREA * random() - SAREA / 2.0,
            SAREA * random() - SAREA / 2.0]


# Simulation parameters
Kp = 2
dt = 0.1
N_ITERATIONS = 10000

# States
WAIT_FOR_NEW_GOAL = 1
MOVING_TO_GOAL = 2

save_dir = './04_output/pngs/'
save_fname_prefix = 'point_control'

# ----------
# link_lengths = [1] * N_LINKS
# joint_angles = np.array([0] * N_LINKS)

link_lengths = [0.5, 1.5, 1.8, 1.2, 0.5]
joint_angles = np.array([1.6, -1.6, 0.8, -0.5, 0.6])

N_LINKS = len(link_lengths)

goal_pos = get_random_goal()

print(f'link length : {link_lengths}')
print(f'joint_angles : {joint_angles}')
print(f'goal : {goal_pos}')


# ----------
arm = NLinkArm(link_lengths, joint_angles, goal_pos)
goal_pos = np.array(arm.goal)

errors, distance = distance_to_goal(arm.end_effector, goal_pos)

joint_goal_angles, solution_found = inverse_kinematics(
    link_lengths, joint_angles, goal_pos)

print(f'### solution found : {solution_found}')

cnt = 0
save_path = save_dir + save_fname_prefix + f'_{str(cnt).zfill(3)}.png'
if solution_found:
    plot_point_control(arm, save_path=save_path)

if distance > 0.1 and solution_found:
    while distance > 0.1:
        cnt += 1
        save_path = save_dir + save_fname_prefix + f'_{str(cnt).zfill(3)}.png'
        # ----------
        # update
        joint_angles = joint_angles + Kp * ang_diff(joint_goal_angles, joint_angles) * dt
        # ----------
        arm.update_joints(joint_angles)
        errors, distance = distance_to_goal(arm.end_effector, goal_pos)
        print(f'distance : {np.round(distance, 3)}')
        plot_point_control(arm, save_path=save_path)
