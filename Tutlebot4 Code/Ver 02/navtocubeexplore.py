#!/usr/bin/env python3
"""
Enhanced cube seeker for TurtleBot4:
 1. Record the robot’s start position as “home.”
 2. Wait for Nav2 to come online.
 3. Execute exploration to build a complete map.
 4. Perform forward‐peek scanning for the cube.
 5. Back‐project detected pixel + depth → map‐frame goal.
 6. Drive to the cube and return home.
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

        # Nav2 + TF + simple motion helper
        self.nav = TurtleBot4Navigator()

        # HSV bounds for yellow/golden cube (tweak as needed)
        self.hsv_min = np.array([15, 100, 100])
        self.hsv_max = np.array([45, 255, 255])

        # Bridge for ROS Image ↔ OpenCV
        self.bridge = CvBridge()

        # Buffers for sensor data
        self.latest_color = None
        self.latest_depth = None
        self.cam_info     = None

        # Subscribe to camera and info topics
        self.create_subscription(Image, '/camera/color/image_raw',              self.on_color,    10)
        self.create_subscription(Image, '/camera/aligned_depth_to_color/image_raw', self.on_depth,    10)
        self.create_subscription(CameraInfo, '/camera/color/camera_info',        self.on_cam_info, 1)

        # Start main sequence
        self.create_timer(1.0, self.main_sequence, once=True)
        self.get_logger().info("CubeSeeker initializing...")

    def on_color(self, msg: Image):        self.latest_color = msg
    def on_depth(self, msg: Image):        self.latest_depth = msg
    def on_cam_info(self, msg: CameraInfo):
        if self.cam_info is None:
            self.cam_info = msg

    def main_sequence(self):
        # 1) Record start as home
        home = self.nav.getCurrentPose()
        self.get_logger().info(
            f"Home recorded at x={home.pose.position.x:.2f}, y={home.pose.position.y:.2f}"
        )

        # 2) Wait for Nav2
        self.get_logger().info("Waiting for Nav2 to activate...")
        self.nav.waitUntilNav2Active()

        # 3) Explore environment to build map
        self.get_logger().info("Starting exploration to build map...")
        try:
            # uses Nav2’s exploration behavior (must be configured in BT)
            self.nav.explore()  
        except AttributeError:
            # fallback: roam random waypoints manually
            self.manual_explore(steps=4, distance=1.0)
        self.get_logger().info("Exploration complete. Map ready.")

        # 4) Forward‐peek scan for cube
        self.get_logger().info("Scanning for cube via forward steps...")
        scan = self.scan_forward_for_cube(step_size=0.3, max_steps=10, max_depth=5.0)
        if scan is None:
            self.get_logger().error("Cube not found. Aborting.")
            rclpy.shutdown(); return
        u, v, depth_m = scan
        self.get_logger().info(
            f"Cube detected at pixel (u={u}, v={v}) depth={depth_m:.2f} m"
        )

        # 5) Compute map‐frame goal
        goal = self.compute_map_goal(u, v)

        # 6) Drive to cube and back
        self.get_logger().info("Driving to cube...")
        self.nav.startToPose(goal)
        self.get_logger().info("Returning home...")
        self.nav.startToPose(home)

        self.get_logger().info("Mission finished.")
        rclpy.shutdown()

    def manual_explore(self, steps: int, distance: float):
        """
        Fallback exploration: move to cardinal waypoints around start.
        """
        for angle in np.linspace(0, 2*np.pi, steps, endpoint=False):
            # rotate
            yaw_goal = self.nav.getPoseStamped([0, 0], angle)
            self.nav.startToPose(yaw_goal)
            # forward
            self.nav.translateDistance(distance)

    def scan_forward_for_cube(self, step_size: float, max_steps: int, max_depth: float):
        """
        Move forward until cube is in view and within depth range.
        """
        for i in range(max_steps):
            self.get_logger().info(f"Step {i+1}: forward {step_size} m...")
            self.nav.translateDistance(step_size)
            rclpy.spin_once(self, timeout_sec=0.5)
            if None in (self.latest_color, self.latest_depth, self.cam_info):
                continue
            img = self.bridge.imgmsg_to_cv2(self.latest_color, 'bgr8')
            uv = self.find_uv(img)
            if not uv: continue
            u, v = uv
            depth_img = self.bridge.imgmsg_to_cv2(self.latest_depth, 'passthrough')
            depth_m = depth_img[v, u] * 0.001
            if 0.1 < depth_m < max_depth:
                return u, v, depth_m
            self.get_logger().info(f"Seen but out of range: {depth_m:.2f} m")
        return None

    def find_uv(self, img: np.ndarray):
        hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_min, self.hsv_max)
        cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: return None
        c = max(cnts, key=cv2.contourArea)
        M = cv2.moments(c)
        if M['m00']==0: return None
        return int(M['m10']/M['m00']), int(M['m01']/M['m00'])

    def compute_map_goal(self, u: int, v: int):
        depth_img = self.bridge.imgmsg_to_cv2(self.latest_depth, 'passthrough')
        z = depth_img[v, u] * 0.001
        k = self.cam_info.k
        x = (u - k[2]) * z / k[0]
        y = (v - k[5]) * z / k[4]
        return self.nav.transformPointToMap([x, y, z])


def main():
    rclpy.init()
    CubeSeeker()
    rclpy.spin()

if __name__ == '__main__':
    main()
