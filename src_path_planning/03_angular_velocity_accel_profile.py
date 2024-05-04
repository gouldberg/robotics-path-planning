# Velocity, Acceleration Profile
# Here Trapezoid pattern

from math import pi
import numpy as np




# ----------
# secs
total_time = 1.0
accel_time = 0.01

# degree
angle_to_move = 90.


# ----------
# compute

# deg / sec
angle_vel_max = angle_to_move / (total_time - accel_time)

# deg / sec**2
angle_accel = angle_vel_max / accel_time

# deg / sec
motor_angle_vel = angle_to_move * 6 / 0.229

# deg / sec**2
motor_angle_accel = angle_accel / 2140.577


print(f'angle vel max     : {angle_vel_max} [deg / sec]')
print(f'angle accel       : {angle_accel} [deg / sec2]')
print(f'motor angle vel   : {motor_angle_vel} [deg / sec]')
print(f'motor angle accel : {motor_angle_accel} [deg / sec2]')
