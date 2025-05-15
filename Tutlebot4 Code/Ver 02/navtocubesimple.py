#!/usr/bin/env python3
"""
Simplified Cube Seeker for TurtleBot4 (ROS 2)

Workflow:
 1. record_home(): Save the starting pose.
 2. ensure_nav_ready(): Wait for Nav2 to be active.
 3. explore_map(): Optional exploration to build the map.
 4. find_cube(): Move forward in steps until cube is detected with valid depth.
 5. compute_goal_pose(u, v): Convert pixel and depth to map-frame pose.
 6. navigate_to(goal) and return_home(): Drive to cube and back.
"""

import rclpy
from rclpy.node import Node
from turtlebot4_python_tutorials.navigator import TurtleBot4Navigator
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np

class CubeSeeker(Node):
    def __init__(self):
        super().__init__('cube_seeker')
        self.nav = TurtleBot4Navigator()
        self.bridge = CvBridge()

        # HSV color thresholds for the cube
        self.hsv_min = np.array([15, 100, 100])
        self.hsv_max = np.array([45, 255, 255])

        # Buffers for sensor data
        self.color_image = None
        self.depth_image = None
        self.camera_info = None

        # Subscribe to camera topics
        self.create_subscription(Image, '/camera/color/image_raw', self.on_color, 10)
        self.create_subscription(Image, '/camera/aligned_depth_to_color/image_raw', self.on_depth, 10)
        self.create_subscription(CameraInfo, '/camera/color/camera_info', self.on_camera_info, 1)

        # Start main process
        self.create_timer(1.0, self.main, once=True)
        self.get_logger().info('CubeSeeker initialized.')

    # Callbacks
    def on_color(self, msg: Image):      self.color_image = msg
    def on_depth(self, msg: Image):      self.depth_image = msg
    def on_camera_info(self, msg: CameraInfo):
        if self.camera_info is None:
            self.camera_info = msg

    def record_home(self):
        pose = self.nav.getCurrentPose()
        self.get_logger().info(f'Home at x={pose.pose.position.x:.2f}, y={pose.pose.position.y:.2f}')
        return pose

    def ensure_nav_ready(self):
        self.get_logger().info('Waiting for Nav2...')
        self.nav.waitUntilNav2Active()

    def explore_map(self):
        try:
            self.nav.explore()
            self.get_logger().info('Exploration complete.')
        except AttributeError:
            self.get_logger().info('No explore() available; skipping.')

    def find_cube(self, step=0.3, max_steps=10, max_depth=5.0):
        for i in range(max_steps):
            self.get_logger().info(f'Step {i+1}: forward {step} m')
            self.nav.translateDistance(step)
            rclpy.spin_once(self, timeout_sec=0.5)

            if not all([self.color_image, self.depth_image, self.camera_info]):
                continue

            u, v = self.detect_uv(self.color_image)
            if u is None:
                continue

            depth = self.get_depth(u, v)
            if 0.1 < depth < max_depth:
                self.get_logger().info(f'Cube at ({u},{v}), depth {depth:.2f} m')
                return u, v

            self.get_logger().info(f'Detected but out of range ({depth:.2f} m)')
        return None, None

    def detect_uv(self, img_msg):
        img = self.bridge.imgmsg_to_cv2(img_msg, 'bgr8')
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_min, self.hsv_max)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None, None
        c = max(cnts, key=cv2.contourArea)
        M = cv2.moments(c)
        if M['m00'] == 0:
            return None, None
        return int(M['m10']/M['m00']), int(M['m01']/M['m00'])

    def get_depth(self, u, v):
        depth_img = self.bridge.imgmsg_to_cv2(self.depth_image, 'passthrough')
        return depth_img[v, u] * 0.001

    def compute_goal_pose(self, u, v):
        depth = self.get_depth(u, v)
        k = self.camera_info.k
        x = (u - k[2]) * depth / k[0]
        y = (v - k[5]) * depth / k[4]
        return self.nav.transformPointToMap([x, y, depth])

    def navigate_to(self, pose):
        self.nav.startToPose(pose)

    def return_home(self, home_pose):
        self.navigate_to(home_pose)

    def main(self):
        home = self.record_home()
        self.ensure_nav_ready()
        self.explore_map()

        u, v = self.find_cube()
        if u is None:
            self.get_logger().error('Cube not found.')
            rclpy.shutdown()
            return

        goal = self.compute_goal_pose(u, v)
        self.navigate_to(goal)
        self.return_home(home)
        self.get_logger().info('Done!')
        rclpy.shutdown()


def main():
    rclpy.init()
    node = CubeSeeker()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
