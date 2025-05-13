import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32
from mvp_msgs.msg import ControlProcess
from std_srvs.srv import SetBool

class SetPointPublisher(Node):
    def __init__(self):
        super().__init__('set_point_publisher')

        self.publisher_ = self.create_publisher(ControlProcess, '/mvp2_test_robot/controller/process/set_point', 10)

        self.interval = 100
        self.set_depth = [-5.0, -5.0,  -5.0, -2.50,  -5.0,  -5.0, 0.0]
        self.set_u =     [ 0.0,  0.3,   0.3,  0.3,    0.0,   0.0, 0.0]
        self.set_yaw =   [3.14, 3.14,   0.0,  0.0,    3.14,  0.0, 0.0]

        #initial set point
        self.set_point = ControlProcess()
        self.set_point.orientation.x = 0.0
        self.set_point.orientation.y = 0.0
        self.set_point.velocity.y = 0.0
        self.set_point.header.frame_id = "mvp2_test_robot/world"
        self.set_point.child_frame_id = "mvp2_test_robot/base_link"
        self.set_point.control_mode = "4dof"
        self.set_point.position.z =self.set_depth[0]
        self.set_point.orientation.z = self.set_yaw[0]
        self.set_point.velocity.x = self.set_u[0]
        self.set_counter = 0  # Start from 0
        
        self.set_controller = self.create_client(SetBool, '/race2_auv/controller/set')  
        self.active_controller(True)

        self.timer_set_update = self.create_timer(100.0, self.timer_update_callback)
        self.timer_pub = self.create_timer(0.5, self.timer_pub)

    def active_controller(self, req: bool):
        set_controller = SetBool.Request()
        set_controller.data = req
        future = self.set_controller.call_async(set_controller)
        # rclpy.spin_until_future_complete(self, future)
        return future.result()


    def timer_update_callback(self):
        if (self.set_counter<len(self.set_depth)):
            self.set_point.position.z =self.set_depth[self.set_counter]
            self.set_point.orientation.z = self.set_yaw[self.set_counter]
            self.set_point.velocity.x = self.set_u[self.set_counter]
            self.get_logger().info(f'Publishing set point number {self.set_counter}')
            self.set_counter +=1

            # set_point = ControlProcess()
            # set_point.orientation.x = 0.001
            # set_point.orientation.y = 0.001
            # set_point.velocity.y = 0.001
            # set_point.header.frame_id = "race2_auv/world"
            # set_point.child_frame_id = "race2_auv/base_link"
            # set_point.control_mode = "4dof"
            # set_point.position.z = 0.01
            # set_point.orientation.z = 0.01
            # set_point.velocity.x = 0.01
            # self.publisher_.publish(set_point)
            
        else:
            self.get_logger().info(f'Setpoint complete, please stop this node')
            self.active_controller(False)


    def timer_pub(self):
        if (self.set_counter<len(self.set_depth)):
            self.publisher_.publish(self.set_point)



def main(args=None):
    rclpy.init(args=args)
    node = SetPointPublisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
