#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import LaserScan, Image
from nav_msgs.msg import Odometry
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from cv_bridge import CvBridge
import cv2
import numpy as np

class TurtleBot4Navigator(Node):
    def __init__(self):
        super().__init__('turtlebot4_navigator')
        
        # Initialize TurtleBot4-specific components from tutorials
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.camera_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        
        # From TurtleBot4 tutorials: TF2 for coordinate transforms
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        self.bridge = CvBridge()
        self.timer = self.create_timer(0.1, self.navigation_loop)
        
        # State variables
        self.current_pose = None
        self.obstacle_detected = False
        self.cube_found = False

    def odom_callback(self, msg):
        """Improved odometry handling with TF2 (from tutorials)"""
        try:
            transform = self.tf_buffer.lookup_transform(
                'odom',
                'base_footprint',
                rclpy.time.Time())
            self.current_pose = transform.transform
        except TransformException as ex:
            self.get_logger().warn(f'TF2 error: {ex}')

    def scan_callback(self, msg):
        """Enhanced obstacle detection (aligned with TurtleBot4 specs)"""
        front_angle_range = 60  # Degrees (30 each side)
        front_ranges = list(msg.ranges[:30]) + list(msg.ranges[-30:])
        self.obstacle_detected = min(front_ranges) < 0.4  # 0.4m threshold

    def image_callback(self, msg):
        """Cube detection using official TurtleBot4 camera specs"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
            
            # Golden color range (calibrate for your lighting)
            lower_gold = np.array([15, 50, 50])
            upper_gold = np.array([45, 255, 255])
            
            mask = cv2.inRange(hsv, lower_gold, upper_gold)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 1000:  # Filter small noise
                    x, y, w, h = cv2.boundingRect(cnt)
                    aspect_ratio = float(w)/h
                    if 0.8 < aspect_ratio < 1.2:  # Square-like shape
                        self.cube_found = True
                        self.get_logger().info("Golden cube detected!")
        except Exception as e:
            self.get_logger().error(f"Camera error: {str(e)}")

    def navigate_to_goal(self, goal_pose):
        """Basic proportional controller for navigation (from tutorials)"""
        twist = Twist()
        if self.current_pose:
            # Calculate errors
            dx = goal_pose.position.x - self.current_pose.translation.x
            dy = goal_pose.position.y - self.current_pose.translation.y
            distance = np.sqrt(dx**2 + dy**2)
            
            # P-controller
            twist.linear.x = 0.5 * distance
            twist.angular.z = 1.0 * np.arctan2(dy, dx)
            
            # Stop if close enough
            if distance < 0.1:
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                return True
        self.cmd_vel_pub.publish(twist)
        return False

    def navigation_loop(self):
        """State machine using TurtleBot4 best practices"""
        if not self.cube_found:
            if self.obstacle_detected:
                # Obstacle avoidance maneuver
                twist = Twist()
                twist.angular.z = 0.5  # Turn right
                self.cmd_vel_pub.publish(twist)
            else:
                # Explore
                twist = Twist()
                twist.linear.x = 0.2
                self.cmd_vel_pub.publish(twist)
        else:
            # Return to origin (simplified)
            goal = PoseStamped()
            goal.pose.position.x = 0.0
            goal.pose.position.y = 0.0
            if self.navigate_to_goal(goal.pose):
                self.get_logger().info("Mission complete!")
                rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    navigator = TurtleBot4Navigator()
    rclpy.spin(navigator)
    navigator.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()