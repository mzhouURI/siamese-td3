import time
import numpy as np
import rclpy
from rclpy.node import Node
from mvp_msgs.msg import ControlProcess
from std_srvs.srv import SetBool
import random
import numpy as np
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Float64, Float64MultiArray
import torch
import collections


class MPCROS(Node):
    def __init__(self):
        super().__init__('ddpg_node')
        self.set_point_interval = 100
        #initial set point
        
        self.subscription = self.create_subscription(ControlProcess, '/mvp2_test_robot/controller/process/value', self.state_callback, 1)
        self.subscription2 = self.create_subscription(ControlProcess, '/mvp2_test_robot/controller/process/error', self.state_error_callback, 1)

        self.set_point_pub = self.create_publisher(ControlProcess, '/mvp2_test_robot/controller/process/set_point', 3)
        self.thruster_pub = self.create_publisher(Float64MultiArray, '/mvp2_test_robot/stonefish/thruster_command', 5)


        self.loss_pub = self.create_publisher(Float64MultiArray, '/training/loss',10)
        self.total_reward_pub = self.create_publisher(Float64, '/training/episode_reward', 10)
                 #initial set point
        self.set_point = ControlProcess()
        self.set_point.orientation.x = 0.0
        self.set_point.orientation.y = 0.0
        self.set_point.velocity.y = 0.0
        self.set_point.header.frame_id = "mvp2_test_robot/world"
        self.set_point.child_frame_id = "mvp2_test_robot/base_link"
        self.set_point.control_mode = "4dof"
        self.set_point.position.z = 0.0
        self.set_point.orientation.z = 0.0
        self.set_point.velocity.x = 0.0
        self.set_counter = 1  # Start from 0

        self.thrust_cmd = [0,0]
        ##setting mode
        self.training = True
        self.training_episode = 0
        self.batch_size = 64
        self.batch_warmup_size =self.batch_size*1
        self.set_point_update_flag = False
        state = {
            'z': (0),
            'euler': (0,0,0,0,0,0), # Quaternion (x, y, z, w)
            'uvw': (0,0,0),
            'pqr': (0,0,0),
        }
        error_state = {
             'z': (0),
            'euler': (0,0,0,0), # Quaternion (x, y, z, w)
            'u': (0)
        }
        self.state = self.flatten_state(state)
        self.error_state = self.flatten_state(error_state)

        self.prev_action = torch.zeros(4)
        self.prev_error_state = torch.zeros(len(self.error_state))
        self.prev_state = torch.zeros(len(self.state))
        self.window_size = 20

        self.state_buffer = collections.deque(maxlen=self.window_size)
        self.error_state_buffer = collections.deque(maxlen=self.window_size)


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = RL_MPC_Agent(state_dim = len(self.state), error_dim = len(self.error_state), action_dim = 4,
                                hidden_dim = 128, num_layers = 2, device = self.device,
                                actor_ckpt = 'actor_transformer.pth',
                                actor_lr = 1e-7, critic_lr= 1e-3, policy_delay=10,
                                seq_len = self.window_size)

        
        self.total_reward = 0

        self.timer_setpoint_update = self.create_timer(100, self.set_point_update)
        self.timer_setpoint_pub = self.create_timer(1.0, self.set_point_publish)
        self.timer_pub = self.create_timer(0.1, self.step)

        self.set_controller = self.create_client(SetBool, '/mvp2_test_robot/controller/set')  
        self.active_controller(True)