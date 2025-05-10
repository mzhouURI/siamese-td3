import time
import numpy as np
from agent import SACAgent
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

class SAC_ROS(Node):
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
        self.batch_size = 64
        self.batch_warmup_size =self.batch_size*1
        self.set_point_update_flag = False
        state = {
            # 'z': (0),
            'euler': (0,0,0), # Quaternion (x, y, z, w)
            'uvw': (0,0,0),
            'pqr': (0,0,0),
        }
        # state = {
        #     #  'z': (0),
        #     'euler': (0,0), # Quaternion (x, y, z, w)
        #     'u': (0, 0),
        #     'pqr': (0)
        # }
        error_state = {
             'z': (0),
            'euler': (0,0), # Quaternion (x, y, z, w)
            'u': (0)
        }
        self.state = self.flatten_state(state)
        self.error_state = self.flatten_state(error_state)
        self.prev_action = torch.zeros(4)
        self.prev_error_state = torch.zeros(len(self.error_state))
        self.prev_state = torch.zeros(len(self.state))
        self.window_size = 20
        self.integral_error_size = 1000

        self.state_buffer = collections.deque(maxlen=self.window_size)
        self.error_state_buffer = collections.deque(maxlen=self.window_size)
        self.integral_error_buffer = collections.deque(maxlen=self.integral_error_size)


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SACAgent(obs_dim = len(self.state) + len(self.error_state), action_dim = 4,
                                seq_len = self.window_size, batch_size = self.batch_size,
                                device = self.device,
                                hidden_dim = 128, num_layers = 2,
                                actor_ckpt = 'actor_rnn.pth',
                                actor_lr = 1e-7, critic_lr= 1e-3,  tau = 0.002
                                )

        self.hidden = self.model.init_hidden(self.batch_size)  # e.g., (h, c) for LSTM

        self.total_reward = 0

        self.timer_setpoint_update = self.create_timer(100, self.set_point_update)
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
            # 'z': (msg.position.z),
            'euler': (msg.orientation.x, msg.orientation.y, msg.orientation.z), 
            'uvw': (msg.velocity.x, msg.velocity.y, msg.velocity.z),
            'pqr': {msg.angular_rate.x, msg.angular_rate.y, msg.angular_rate.z}
        }
        # state = {
        #     # 'z': (msg.position.z),
        #     'euler': (msg.orientation.y, msg.orientation.z),  
        #     'u': (msg.velocity.x, msg.velocity.z),
        #     'pqr': (msg.angular_rate.z)
        # }
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

            # Convert to torch tensor
            # new_state = torch.tensor(self.state + self.error_state, dtype=torch.float32)
            new_state = torch.tensor(np.concatenate([self.state, self.error_state]), dtype=torch.float32)
            new_error_state = torch.tensor(self.error_state, dtype=torch.float32).unsqueeze(0)
            
            self.state_buffer.append(new_state)
            self.integral_error_buffer.append(new_error_state)

            # do RNN inference when there are enough states
            if self.set_point_update_flag:
                    self.set_point_update_flag = False
                    print("skip this step")
                    self.state_buffer.clear()
                    self.error_state_buffer.clear()
                    self.integral_error_buffer.clear()
                
            if(len(self.state_buffer)==self.window_size):
                # state_seq = torch.cat(list(self.state_buffer), dim=0).unsqueeze(0).to(self.device)

                action, self.hidden = self.model.actor(new_state, self.hidden)

                msg = Float64MultiArray()
                msg.data = action.detach().cpu().numpy().flatten().tolist()                   
                self.thruster_pub.publish(msg)

                reward = 0
                #error state reward
                done = False

                ##detect set point changge, and clear the temporal buffer.        
                # Convert the deque to a PyTorch tensor
                buffer_integral_error = torch.stack(list(self.integral_error_buffer))
                reward = self.calculate_reward(self.prev_error_state, new_error_state, buffer_integral_error)

                self.model.replay_buffer.add(self.prev_state, self.prev_action.detach().cpu().numpy(), 
                                                reward, new_state, new_error_state, done)
                    
                self.prev_error_state = new_error_state
                self.prev_state = new_state
                self.prev_action = action

                self.total_reward  = self.total_reward + reward

            if len(self.model.replay_buffer.buffer) > self.batch_warmup_size:  # Start training after enough experiences
                c1_loss, c2_loss, actor_loss = self.model.train(batch_size=self.batch_size)
                msg = Float64MultiArray()
                msg.data = [float(c1_loss), float(c2_loss), float(actor_loss)]
                self.loss_pub.publish(msg)
        # except Exception as e:
        #     self.get_logger().error(f"Error in step(): {e}")

    def calculate_reward(self, error_pose, new_error_pose, histo_error):
        new_error = new_error_pose.view(-1, 1)
        current_error = error_pose.view(-1,1)
        histo_error = histo_error.squeeze(1)

        # Define a weight matrix W: shape [3, 3]
        w_z = 0.25
        w_pitch = 0.25
        w_yaw = 0.25
        w_u = 0.25
        
        # Creat diagonal weight matrix W
        W = torch.tensor([w_z, w_pitch, w_yaw, w_u], dtype=torch.float32)
        error_reward = torch.sum(abs(new_error) * W )

        w_z = 0.25
        w_pitch = 0.25
        w_yaw = 0.25
        w_u = 0.25

        W = torch.tensor([w_z, w_pitch, w_yaw, w_u], dtype=torch.float32, device = histo_error.device)
        # sum all the error and calcualte the mean
        histo_error =  torch.sum(histo_error, dim=0)
        # print(histo_error)
        accum_error = torch.sum(abs(histo_error) * W )

        w_d_z = 0.25
        w_d_pitch = 0.25
        w_d_yaw = 0.25
        w_d_u = 0.25
        weights = torch.tensor([w_d_z, w_d_pitch, w_d_yaw, w_d_u], dtype=torch.float32)


        current_weighted = current_error * weights
        new_weighted = new_error * weights
        delta_pose = abs(new_weighted) - abs(current_weighted)
        delta_reward =  torch.sum (delta_pose )
 
        bonus = -1000
        if abs(new_error[0])<0.1 and abs(new_error[1])<0.05 and abs(new_error[2])<0.05 and abs(new_error[3])<0.01:
            bonus = 0
            # print("bonus")

        error_reward = -100*error_reward
        delta_reward = -10000*delta_reward
        accum_error = -1*accum_error
        reward = error_reward +delta_reward + accum_error + bonus
        print(f"error_reward: {error_reward: .4f}",
              f"accum_error: {accum_error: .4f}",
              f"delta_reward: {delta_reward: .4f}",
              f"bonus: {bonus: .4f}")

        return reward




def main(args=None):

    rclpy.init(args=args)
    node = SAC_ROS()
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
