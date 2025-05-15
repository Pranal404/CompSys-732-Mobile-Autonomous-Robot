#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import LaserScan, Image
from nav_msgs.msg import Odometry
from tf2_ros import Buffer, TransformListener
from cv_bridge import CvBridge
import cv2
import numpy as np

class SimpleNavigator(Node):
    """
    A streamlined navigator for TurtleBot4:
    - Records start position
    - Avoids obstacles using laser scan
    - Detects a yellow cube with camera
    - Returns home after finding cube
    """
    def __init__(self):
        super().__init__('simple_navigator')
        # Publishers & subscribers
        self.vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)
        self.create_subscription(Image, '/camera/color/image_raw', self.image_cb, 10)

        # TF2 buffer (for future use if needed)
        self.tf_buffer = Buffer()
        TransformListener(self.tf_buffer, self)

        # Image bridge
        self.bridge = CvBridge()

        # State variables
        self.pose      = None   # Current pose from odometry
        self.obstacle  = False  # True if obstacle < 0.4m ahead
        self.cube_seen = False  # True once cube is detected
        self.home      = None   # Start pose to return to

        # Main loop timer
        self.create_timer(0.1, self.main_loop)
        self.get_logger().info('SimpleNavigator started')

    def odom_cb(self, msg: Odometry):
        # Store the latest pose
        self.pose = msg.pose.pose

    def scan_cb(self, msg: LaserScan):
        # Check front 60° for obstacles
        front = msg.ranges[:30] + msg.ranges[-30:]
        if front:
            self.obstacle = min(front) < 0.4

    def image_cb(self, msg: Image):
        # Detect yellow cube in color frame
        img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (15,50,50), (45,255,255))
        cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Mark cube seen if a sufficiently large blob appears
        if cnts and max(cv2.contourArea(c) for c in cnts) > 1000:
            self.cube_seen = True
            self.get_logger().info('Cube detected')

    def main_loop(self):
        # Record home pose once
        if self.home is None and self.pose:
            self.home = PoseStamped()
            self.home.header.frame_id = 'odom'
            self.home.pose = self.pose
            self.get_logger().info(f'Home saved at x={self.pose.position.x:.2f}, y={self.pose.position.y:.2f}')

        # Before cube: explore and avoid obstacles
        if not self.cube_seen:
            cmd = Twist()
            if self.obstacle:
                # Turn in place if blocked
                cmd.angular.z = 0.5
            else:
                # Move forward otherwise
                cmd.linear.x = 0.2
            self.vel_pub.publish(cmd)
        else:
            # After seeing cube: return home
            if self.home and self.navigate_to(self.home):
                self.get_logger().info('Returned home. Task complete.')
                rclpy.shutdown()

    def navigate_to(self, target: PoseStamped) -> bool:
        """
        Simple P-controller to move toward a target pose.
        Returns True when within 0.1m.
        """
        if not self.pose:
            return False
        dx = target.pose.position.x - self.pose.position.x
        dy = target.pose.position.y - self.pose.position.y
        dist = np.hypot(dx, dy)

        cmd = Twist()
        if dist > 0.1:
            # Linear speed proportional to distance
            cmd.linear.x = 0.5 * dist
            # Angular speed toward heading
            cmd.angular.z = 2.0 * np.arctan2(dy, dx)
            self.vel_pub.publish(cmd)
            return False
        return True


def main():
    rclpy.init()
    nav = SimpleNavigator()
    rclpy.spin(nav)
    nav.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
