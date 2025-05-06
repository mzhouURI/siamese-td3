from read_topics import read_controller_values, read_thrusters, read_joints
import matplotlib.pyplot as plt
import rosbag2_py
from utils import inteporlate_state, inteporlate_thruster
import numpy as np

f_name = "rosbag2_2025_05_05-18_34_08" #+-135

bag_name = f_name + "_0.mcap"
BAG_PATH =  f_name + "/" + f_name + "_0.mcap" 

# Odometry topic
NAME_SPACE = '/mvp2_test_robot'
C_TOPIC = '/controller/process/set_point'
M_TOPIC = '/controller/process/value'
E_TOPIC = '/controller/process/error'

thruster_topics = ['/control/thruster/surge',
                '/control/thruster/sway_bow',
                '/control/thruster/heave_stern',
                '/control/thruster/heave_bow']         

# Read data
set_points, states, state_errors = read_controller_values(BAG_PATH, NAME_SPACE, C_TOPIC, M_TOPIC, E_TOPIC)

print(set_points.shape)
thruster_t, thruster_data = read_thrusters(BAG_PATH, NAME_SPACE, thruster_topics)

print(thruster_t.shape)

##interpolate set_points to state_errors time
reftime = state_errors[:,0]

new_states = inteporlate_state(states, reftime, kind = 'linear')
new_set_points = inteporlate_state(set_points, reftime, kind = 'previous')

new_thruster_data = inteporlate_thruster(thruster_t, thruster_data, reftime, kind = 'previous')

combined_up = np.hstack((state_errors, new_states, new_set_points, new_thruster_data))

np.savetxt('filename1.csv', combined_up, delimiter=',')