#!/usr/bin/env python3
"""
Simple cube seeker for TurtleBot4 using Nav2 + TB4 helper with forward‐peek scanning.

Steps:
 1. Record the robot’s current position as “home” at startup.
 2. Wait for Nav2 to come online.
 3. Move forward in small increments until the cube appears in the RGB image and depth is valid.
 4. Back‐project pixel + depth → a map‑frame goal pose.
 5. Drive to the cube.
 6. Return to the recorded home pose.
"""

import rclpy
from rclpy.node import Node

# Helper for Nav2, TF, docking/undocking, and simple motions
from turtlebot4_python_tutorials.navigator import TurtleBot4Navigator

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np

class CubeSeeker(Node):
    def __init__(self):
        super().__init__('cube_seeker')

        # 2.1) Nav2 + TF + simple motion helper
        self.nav = TurtleBot4Navigator()

        # 2.2) HSV bounds for yellow/golden cube (tweak as needed)
        self.hsv_min = np.array([15, 100, 100])
        self.hsv_max = np.array([45, 255, 255])

        # 2.3) Bridge for ROS Image ↔ OpenCV
        self.bridge = CvBridge()

        # 2.4) Buffers for latest sensor messages
        self.latest_color = None
        self.latest_depth = None
        self.cam_info     = None

        # 3) Subscribe to camera topics
        self.create_subscription(Image, '/camera/color/image_raw',              self.on_color,    10)
        self.create_subscription(Image, '/camera/aligned_depth_to_color/image_raw', self.on_depth,    10)
        self.create_subscription(CameraInfo, '/camera/color/camera_info',        self.on_cam_info, 1)

        # 4) Kick off main sequence after a short delay
        self.create_timer(1.0, self.main_sequence, once=True)
        self.get_logger().info("CubeSeeker initialized, starting soon...")

    # ────────────────────────────────────────────────────────────────────────────
    # Callbacks to store latest images & intrinsics
    def on_color(self, msg: Image):        self.latest_color = msg
    def on_depth(self, msg: Image):        self.latest_depth = msg
    def on_cam_info(self, msg: CameraInfo):
        if self.cam_info is None:
            self.cam_info = msg  # store once

    # ────────────────────────────────────────────────────────────────────────────
    def main_sequence(self):
        # 1) Record current pose as home
        home = self.nav.getCurrentPose()
        self.get_logger().info(
            f"Home recorded at x={home.pose.position.x:.2f}, y={home.pose.position.y:.2f}")

        # 2) Wait for Nav2 to be ready
        self.get_logger().info("Waiting for Nav2 to become active...")
        self.nav.waitUntilNav2Active()

        # 3) Forward‐peek scan for cube
        self.get_logger().info("Scanning for cube via forward steps...")
        result = self.scan_forward_for_cube(step_size=0.3,max_steps=10,max_depth=5.0)
        if result is None:
            self.get_logger().error("Cube not found within scan range. Aborting.")
            rclpy.shutdown()
            return
        u, v, depth_m = result
        self.get_logger().info(f"Cube detected at pixel (u={u}, v={v}), depth={depth_m:.2f} m" )

        # 4) Compute map-frame goal pose
        goal = self.compute_map_goal(u, v)

        # 5) Navigate to cube and return home
        self.get_logger().info("Driving to cube...")
        self.nav.startToPose(goal)
        self.get_logger().info("Returning to home pose...")
        self.nav.startToPose(home)

        self.get_logger().info("Task complete. Shutting down.")
        rclpy.shutdown()

    # ────────────────────────────────────────────────────────────────────────────
    def scan_forward_for_cube(self, step_size: float, max_steps: int, max_depth: float):
        """
        Move forward in increments until the cube is seen and within depth range.
        Returns (u, v, depth_m) or None.
        """
        for i in range(max_steps):
            self.get_logger().info(f"Step {i+1}: moving forward {step_size} m...")
            self.nav.translateDistance(step_size)  # Nav2 handles obstacles

            # ensure we have fresh data
            rclpy.spin_once(self, timeout_sec=0.5)
            if None in (self.latest_color, self.latest_depth, self.cam_info):
                continue

            # detect in color frame
            cv_img = self.bridge.imgmsg_to_cv2(self.latest_color, 'bgr8')
            uv = self.find_uv(cv_img)
            if not uv:
                continue
            u, v = uv

            # get depth at pixel
            depth_img = self.bridge.imgmsg_to_cv2(self.latest_depth, 'passthrough')
            depth_m = depth_img[v, u] * 0.001  # mm→m
            if 0.1 < depth_m < max_depth:
                return u, v, depth_m

            self.get_logger().info(f"Detected color at ({u},{v}) but depth={depth_m:.2f} m out of range, continuing...")
        return None

    # ────────────────────────────────────────────────────────────────────────────
    def find_uv(self, img: np.ndarray):
        """
        HSV threshold → largest contour centroid (u,v) or None.
        """
        hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_min, self.hsv_max)
        cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        c = max(cnts, key=cv2.contourArea)
        M = cv2.moments(c)
        if M['m00'] == 0:
            return None
        return int(M['m10']/M['m00']), int(M['m01']/M['m00'])

    # ────────────────────────────────────────────────────────────────────────────
    def compute_map_goal(self, u: int, v: int):
        """
        Back-project pixel (u,v) using depth image → 3D camera point → map-frame PoseStamped.
        """
        depth_img = self.bridge.imgmsg_to_cv2(self.latest_depth, 'passthrough')
        z = depth_img[v, u] * 0.001  # mm→m
        k = self.cam_info.k
        x = (u - k[2]) * z / k[0]
        y = (v - k[5]) * z / k[4]
        return self.nav.transformPointToMap([x, y, z])


def main():
    rclpy.init()
    node = CubeSeeker()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
