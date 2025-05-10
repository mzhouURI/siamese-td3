import time
import numpy as np
from agent import TD3Agent
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

class SAC_RNN_ROS(Node):
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

        self.thrust_cmd = [0,0,0,0]
        ##setting mode
        
        self.set_point_update_flag = False
        state = {
            'euler': (0,0,0), # Quaternion (x, y, z, w)
            'uvw': (0,0,0),
            'pqr': (0,0,0),
        }
        error_state = {
             'z': (0),
            'euler': (0,0), # Quaternion (x, y, z, w)
            'u': (0)
        }
        self.state = self.flatten_state(state)
        self.error_state = self.flatten_state(error_state)
        
        self.window_size = 20
        self.batch_size = 64
        self.batch_warmup_size =self.batch_size*1

        self.rnn_obs_buffer = collections.deque(maxlen=self.window_size)
        self.rnn_new_obs_buffer = collections.deque(maxlen=self.window_size)
        self.rnn_action_buffer = collections.deque(maxlen=self.window_size)
        self.rnn_reward_buffer = collections.deque(maxlen=self.window_size)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TD3Agent(obs_dim = len(self.state) + len(self.error_state), action_dim = 4,
                                seq_len = self.window_size,
                                device = self.device,
                                hidden_size = 128, rnn_layer = 2,
                                actor_ckpt = 'actor_rnn.pth',
                                actor_lr = 1e-6, critic_lr= 1e-5,  tau = 0.001,
                                )
        self.total_reward = 0

        self.timer_setpoint_update = self.create_timer(30, self.set_point_update)
        self.timer_setpoint_pub = self.create_timer(1.0, self.set_point_publish)
        self.timer_pub = self.create_timer(0.1, self.step)

        self.set_controller = self.create_client(SetBool, '/mvp2_test_robot/controller/set')  
        self.active_controller(True)
        
    def active_controller(self, req: bool):
        set_controller = SetBool.Request()
        set_controller.data = req
        future = self.set_controller.call_async(set_controller)
        # rclpy.spin_until_action_dimfuture_complete(self, future)
        return future.result()
    
    def set_point_update(self):
        print(f"episode reward = {self.total_reward}")
        self.model.save_model()
        msg = Float64()
        msg.data = float (self.total_reward)
        self.total_reward_pub.publish(msg)
        #update setpoint
        self.set_point.position.z = random.uniform(-5,-1)
        self.set_point.orientation.z = random.uniform(-3.14, 3.14)
        self.set_point.velocity.x = random.uniform(0.0, 0.5)
        self.total_reward = 0
        self.set_point_update_flag = True
    
    def set_point_publish(self):
        self.set_point_pub.publish(self.set_point)

    def wrap_to_pi(self, angle):
        """Wrap angle to [-pi, pi]."""
        return (angle + np.pi) % (2 * np.pi) - np.pi
    
    def flatten_state(self, state_dict):
        def flatten_value(v):
            if isinstance(v, (tuple, list, set, np.ndarray)):
                result = []
                for item in v:
                    result.extend(flatten_value(item))  # recursive call
                return result
            else:
                return [v]

        flat = []
        for value in state_dict.values():
            flat.extend(flatten_value(value))
        
        return np.array(flat, dtype=np.float32)

    def state_callback(self, msg):

        state = {
            'euler': (msg.orientation.x, msg.orientation.y, msg.orientation.z), 
            'uvw': (msg.velocity.x, msg.velocity.y, msg.velocity.z),
            'pqr': {msg.angular_rate.x, msg.angular_rate.y, msg.angular_rate.z}
        }
        self.state = self.flatten_state(state)

    def state_error_callback(self, msg):

        state_error = {
            'z': (msg.position.z),
            'euler': (msg.orientation.y, msg.orientation.z),  
            'u': (msg.velocity.x)
        }
        self.error_state = self.flatten_state(state_error)

    # def save_model(self):

    def step(self):
        # try:
        #if just updated setpoint, we clear the buffer and wait for enough states
        if self.set_point_update_flag:
                self.set_point_update_flag = False
                print("skip this step")
                self.rnn_new_obs_buffer.clear()
                self.rnn_reward_buffer.clear()
                self.rnn_action_buffer.clear()
                reward = 0

        #get the states before update so this is our previous state.
        rnn_prev_obs = self.rnn_new_obs_buffer
        # get new states
        new_state = torch.tensor(np.concatenate([self.state, self.error_state]), dtype=torch.float32)
        self.rnn_new_obs_buffer.append(new_state)

        # get new error state for computing reward
        new_error_state = torch.tensor(self.error_state, dtype=torch.float32).unsqueeze(0)  

        #calculate reward from previous action
        reward = self.calculate_reward(new_error_state)
        reward = reward.detach().float().unsqueeze(0)
        self.rnn_reward_buffer.append(reward)

        #take new action                
        #store the old action seq first
        rnn_prev_actions = self.rnn_action_buffer
        #take new action
        hidden = self.model.actor.init_hidden(self.batch_size)
        obs_seq = torch.stack(list(self.rnn_new_obs_buffer), dim=0)
        # Add batch dimension: shape (1, T, obs_dim)
        obs_seq = obs_seq.unsqueeze(0)
        obs_seq = obs_seq.to(self.device)
        
        action = self.model.select_action(obs_seq, hidden)
        action = action[:, -1, :]  # Shape: (batch_size, state_dim)
        action = action.detach().squeeze()
        # print(action.shape)
        self.rnn_action_buffer.append(action)
        # print(f"rnn action: {self.rnn_action_buffer.shape}")

        #publish action
        msg = Float64MultiArray()
        msg.data = action.detach().cpu().numpy().flatten().tolist()                   
        self.thruster_pub.publish(msg)

        # add to buffer when the rnn sequence buffer is filled
        if(len(self.rnn_action_buffer)==self.window_size):     
            # print("adding")    
            self.model.replay_buffer.add(rnn_prev_obs, rnn_prev_actions, 
                                            self.rnn_reward_buffer, self.rnn_new_obs_buffer)
            self.total_reward  = self.total_reward + reward

        

        #do training if there are enough data in the replay buffer
        if len(self.model.replay_buffer.buffer) > self.batch_warmup_size:  # Start training after enough experiences
            c1_loss, c2_loss, actor_loss = self.model.update(batch_size=self.batch_size)
            msg = Float64MultiArray()
            msg.data = [float(c1_loss), float(c2_loss), float(actor_loss)]
            self.loss_pub.publish(msg)

        
        # except Exception as e:
        #     self.get_logger().error(f"Error in step(): {e}")

    def calculate_reward(self, new_error_pose):
        new_error = new_error_pose.view(-1, 1)

        # Define a weight matrix W: shape [3, 3]
        w_z = 0.25
        w_pitch = 0.25
        w_yaw = 0.25
        w_u = 0.25
        
        # Creat diagonal weight matrix W
        W = torch.tensor([w_z, w_pitch, w_yaw, w_u], dtype=torch.float32)
        error_reward = torch.sum(abs(new_error) * W )

        bonus = -1000
        if abs(new_error[0])<0.1 and abs(new_error[1])<0.05 and abs(new_error[2])<0.05 and abs(new_error[3])<0.01:
            bonus = 0
            # print("bonus")

        error_reward = -100*error_reward
        reward = error_reward + bonus
        print(f"error_reward: {error_reward: .4f}",
              f"bonus: {bonus: .4f}")

        return reward




def main(args=None):

    rclpy.init(args=args)
    node = SAC_RNN_ROS()
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
