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
from network.agent import RL_MPC_Agent
from network.utilites import flatten_state

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
        self.batch_size = 8
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
        self.state = flatten_state(state)
        self.error_state = flatten_state(error_state)

        self.prev_action = torch.zeros(1,4)
        self.prev_error_state = torch.zeros(1,len(self.error_state))
        self.prev_state = torch.zeros(1,len(self.state))
        self.window_size = 100

        self.performance = 0

        self.state_buffer = collections.deque(maxlen=self.window_size)
        self.error_state_buffer = collections.deque(maxlen=self.window_size)


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = RL_MPC_Agent(state_dim = len(self.state), error_dim = len(self.error_state), action_dim = 4,
                                hidden_dim = 256, num_layers = 2, num_head= 8, device = self.device,
                                actor_ckpt = 'offline_model/actor.pth',
                                modeler_ckpt = 'offline_model/modeler.pth',
                                actor_lr = 1e-6, modeler_lr= 1e-6, policy_delay=5,
                                noise_std=0.05, max_action = 0.7,
                                pitch_loss_weight = 10, depth_loss_weight =1, surge_loss_weight =5, yaw_loss_weight =1,
                                smooth_loss_weight =0.1, jerk_loss_weight = 0.1, total_action_weight = 0.1)

        torch.autograd.set_detect_anomaly(True)
        self.timer_setpoint_update = self.create_timer(100, self.set_point_update)
        self.timer_setpoint_pub = self.create_timer(1.0, self.set_point_publish)
        self.timer_pub = self.create_timer(0.1, self.step)

        self.set_controller = self.create_client(SetBool, '/mvp2_test_robot/controller/set')  
        self.active_controller(True)
        # self.model.actor.eval()
        # self.model.modeler.eval()
    

    def active_controller(self, req: bool):
        set_controller = SetBool.Request()
        set_controller.data = req
        future = self.set_controller.call_async(set_controller)
        # rclpy.spin_until_action_dimfuture_complete(self, future)
        return future.result()
    

    def set_point_update(self):
        self.model.save_model()
        print(f"episode reward = {self.performance}")
        self.model.save_model()
        msg = Float64()
        msg.data = float (self.performance)
        self.total_reward_pub.publish(msg)
        #update setpoint
        self.set_point.position.z = random.uniform(-5,-1)
        self.set_point.orientation.z = random.uniform(-3.14, 3.14)
        self.set_point.velocity.x = random.uniform(0.0, 0.5)
        self.total_reward = 0
        self.set_point_update_flag = True

    
    def set_point_publish(self):
        self.set_point_pub.publish(self.set_point)

    def state_callback(self, msg):

        cos_roll = np.cos(msg.orientation.x)
        sin_roll = np.sin(msg.orientation.x)
        cos_pitch = np.cos(msg.orientation.y)
        sin_pitch = np.sin(msg.orientation.y)
        cos_yaw = np.cos(msg.orientation.z)
        sin_yaw = np.sin(msg.orientation.z)
        
        state = {
            'z': (msg.position.z),
            'euler': (cos_roll, sin_roll, cos_pitch, sin_pitch, cos_yaw, sin_yaw), 
            'uvw': (msg.velocity.x, msg.velocity.y, msg.velocity.z),
            'pqr': {msg.angular_rate.x, msg.angular_rate.y, msg.angular_rate.z}
        }
        self.state = flatten_state(state)

    def state_error_callback(self, msg):

        cos_pitch = np.cos(msg.orientation.y)
        sin_pitch = np.sin(msg.orientation.y)
        cos_yaw = np.cos(msg.orientation.z)
        sin_yaw = np.sin(msg.orientation.z)

        state_error = {
            'z': (msg.position.z),
            'euler': (cos_pitch, sin_pitch, cos_yaw, sin_yaw),  
            'u': (msg.velocity.x)
        }
        self.error_state = flatten_state(state_error)

    def step(self):
        new_state = torch.tensor(self.state, dtype=torch.float32).unsqueeze(0)
        new_error_state = torch.tensor(self.error_state, dtype=torch.float32).unsqueeze(0)
        #action
        zero_depth_initial_state = new_state.clone()
        zero_depth_initial_state[:,0] = 0
        actor_states= torch.cat([zero_depth_initial_state, new_error_state], dim = 1)
    
        action = self.model.select_action(actor_states, self.window_size)
        # print(action.shape)
        #pitch the first action from the sequence and command to the vehicle
        msg = Float64MultiArray()
        msg.data = action.detach().cpu().numpy().flatten().tolist()                   
        self.thruster_pub.publish(msg)
        
        self.model.replay_buffer.add(self.prev_state, new_state, self.prev_error_state, new_error_state, self.prev_action.detach().cpu().numpy())
  
        self.prev_state = new_state
        self.prev_error_state = new_error_state
        self.prev_action = action
        
        if len(self.model.replay_buffer.buffer) > self.batch_warmup_size + self.window_size:
            c1_loss, actor_loss = self.model.train(batch_size=self.batch_size, sequence_len = self.window_size)
            msg = Float64MultiArray()
            msg.data = [float(c1_loss), float(actor_loss)]
            self.loss_pub.publish(msg)

        self.performance = self.performance + torch.abs(new_error_state).sum()

   
def main(args=None):

    rclpy.init(args=args)
    node = MPCROS()
    node.set_point_update()

    try:
        rclpy.spin(node)  # This should keep the node alive and run timers and callbacks
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()



if __name__ == '__main__':
    main()
