import time
import numpy as np
from networks import RNNActor
import rclpy
from rclpy.node import Node
from mvp_msgs.msg import ControlProcess
from std_srvs.srv import SetBool
from std_msgs.msg import Bool
import random
import numpy as np
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Float64, Float64MultiArray
import torch
import collections

class ActorROS(Node):
    def __init__(self):
        super().__init__('actor_ros_offline')
        #initial set point
        self.subscription = self.create_subscription(ControlProcess, '/mvp2_test_robot/controller/process/value', self.state_callback, 1)
        self.subscription2 = self.create_subscription(ControlProcess, '/mvp2_test_robot/controller/process/error', self.state_error_callback, 1)
        self.subscription3 = self.create_subscription(Bool, '/mvp2_test_robot/controller/state', self.controller_state_callback, 1)

        self.thruster_pub = self.create_publisher(Float64MultiArray, '/mvp2_test_robot/stonefish/thruster_command', 5)

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

        self.thrust_cmd = []

        state = {
            # 'z': {0},
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
        self.rnn_obs_buffer = collections.deque(maxlen=self.window_size)

        self.current_action = torch.zeros(4, 1)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = self.actor = RNNActor(obs_dim = len(self.state) + len(self.error_state), action_dim = 4, hidden_size = 512, rnn_layers = 3).to(self.device)
        
        # self.model.load_state_dict(torch.load("actor_transformer.pth"))
        self.model.load_state_dict(torch.load("model/actor.pth"))
        self.actor_hidden = self.model.init_hidden(1)

        self.model.eval()

        # Buffer to store the last N actions and states
        self.enabled = False
        #setup timer
        self.timer_pub = self.create_timer(0.1, self.step)
        
    def controller_state_callback(self, msg):
        self.enabled = msg.data
        if not self.enabled:
            action = torch.zeros(4, 1)
            msg = Float64MultiArray()
            msg.data = action.detach().cpu().numpy().flatten().tolist()                   
            self.thruster_pub.publish(msg)
    
    
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
            # 'z': {msg.position.z},
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

    def thruster_callback(self, msg):
        self.thrust_cmd[0] = list(msg.data)

    def step(self):
        if self.enabled :
            new_state = torch.tensor(np.concatenate([self.state, self.error_state]), dtype=torch.float32)
            self.rnn_obs_buffer.append(new_state)
            
            obs_seq = torch.stack(list(self.rnn_obs_buffer), dim=0)
            # Add batch dimension: shape (1, T, obs_dim)
            obs_seq = obs_seq.unsqueeze(0)
            obs_seq = obs_seq.to(self.device)
            
            # Assuming the transformer model takes the sequence of past states and errors as input
            with torch.no_grad():
                action, self.actor_hidden = self.model(obs_seq, hidden = self.actor_hidden)
                self.actor_hidden = tuple(h.detach() for h in self.actor_hidden)
                action = action[:, -1, :]  # Shape: (batch_size, state_dim)
                action = action.detach().squeeze()

                print(action)
                
                # Convert predicted thrust commands to a message
                msg = Float64MultiArray()
                msg.data = action.detach().cpu().numpy().flatten().tolist()                   
                self.thruster_pub.publish(msg)

def main(args=None):

    rclpy.init(args=args)
    node = ActorROS()

    try:
        rclpy.spin(node)  # This should keep the node alive and run timers and callbacks
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()



if __name__ == '__main__':
    main()
