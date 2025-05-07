import time
import numpy as np
from actor import ActorTransformer
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

class ActorROS(Node):
    def __init__(self):
        super().__init__('actor_ros_offline')
        #initial set point
        self.subscription = self.create_subscription(ControlProcess, '/mvp2_test_robot/controller/process/value', self.state_callback, 1)
        self.subscription2 = self.create_subscription(ControlProcess, '/mvp2_test_robot/controller/process/error', self.state_error_callback, 1)

        self.set_point_pub = self.create_publisher(ControlProcess, '/mvp2_test_robot/controller/process/set_point', 3)
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
            'z': (0),
            'euler': (0,0,0), # Quaternion (x, y, z, w)
            'uvw': (0,0,0),
            # 'uvw': (0),
            'pqr': (0,0,0)
        }
        error_state = {
             'z': (0),
            'euler': (0,0), # Quaternion (x, y, z, w)
            'u': (0)
        }
        self.state = self.flatten_state(state)
        self.error_state = self.flatten_state(error_state)

        #setup model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device =device
        print("Using device:", device)

        self.model = ActorTransformer(state_dim = len(self.state), error_dim = len(self.error_state), 
                                hidden_dim = 32, num_layers = 2,
                                output_dim = 4).to(self.device)

        self.model.load_state_dict(torch.load("actor_transformer.pth"))
        self.model.eval()

        #setup sequence buffer
        self.window_size = 49
        # Buffer to store the last N actions and states
        # self.action_buffer = collections.deque(maxlen=self.window_size)
        self.state_buffer = collections.deque(maxlen=self.window_size)
        self.error_state_buffer = collections.deque(maxlen=self.window_size)

        #setup timer
        self.timer_pub = self.create_timer(0.05, self.step)
        self.timer_setpoint_update = self.create_timer(60, self.set_point_update)
        self.timer_setpoint_pub = self.create_timer(1.0, self.set_point_publish)

        self.set_controller = self.create_client(SetBool, '/mvp2_test_robot/controller/set')  
        self.active_controller(True)
        
    def active_controller(self, req: bool):
        set_controller = SetBool.Request()
        set_controller.data = req
        future = self.set_controller.call_async(set_controller)
        # rclpy.spin_until_future_complete(self, future)
        return future.result()
    
    def set_point_update(self):    
        #update setpoint
        self.set_point.position.z = random.uniform(-5,-1)
        self.set_point.orientation.z = random.uniform(-3.14, 3.14)
        self.set_point.velocity.x = random.uniform(0.0, 0.5)
    
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
            'z': (msg.position.z),
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
        current_pose = torch.tensor(self.state, dtype=torch.float32).unsqueeze(0)
        error_pose = torch.tensor(self.error_state, dtype=torch.float32).unsqueeze(0)

        # Add current state and error to the buffers
        self.state_buffer.append(current_pose)
        self.error_state_buffer.append(error_pose)
        
        # Create a sequence of actions (the buffer) for transformer inference
        # The buffer will contain the most recent states and errors
        context_states = torch.cat(list(self.state_buffer), dim=0).unsqueeze(0).to(self.device)
        context_errors = torch.cat(list(self.error_state_buffer), dim=0).unsqueeze(0).to(self.device)
        
        # Assuming the transformer model takes the sequence of past states and errors as input
        with torch.no_grad():
            pred_thrust_cmd = self.model(context_states, context_errors)
            print(pred_thrust_cmd)
            
            # Convert predicted thrust commands to a message
            msg = Float64MultiArray()
            msg.data = pred_thrust_cmd.detach().cpu().numpy().flatten().tolist()
            self.thruster_pub.publish(msg)

def main(args=None):

    rclpy.init(args=args)
    node = ActorROS()
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
