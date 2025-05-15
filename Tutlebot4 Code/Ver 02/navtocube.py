#!/usr/bin/env python3
"""
Simple cube seeker for TurtleBot4 using Nav2 + TB4 helper.

Steps:
 1. Record the robot's current position as "home" when the code starts.
 2. Wait for Nav2 to come online.
 3. Spin in place and look for a yellow cube via HSV threshold.
 4. Convert detected pixel + depth → a map-frame goal pose.
 5. Drive to the cube.
 6. Return to the recorded home pose.
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

        # Nav2 + TF convenience helper
        self.nav = TurtleBot4Navigator()

        # HSV bounds for yellow/golden cube
        self.hsv_min = np.array([15, 100, 100])
        self.hsv_max = np.array([45, 255, 255])

        # Bridge for ROS Image ↔ OpenCV
        self.bridge = CvBridge()

        # Buffers for incoming sensor data
        self.latest_color = None
        self.latest_depth = None
        self.cam_info     = None

        # Subscribe to camera topics
        self.create_subscription(Image, '/camera/color/image_raw', self.on_color, 10)
        self.create_subscription(Image, '/camera/aligned_depth_to_color/image_raw', self.on_depth, 10)
        self.create_subscription(CameraInfo, '/camera/color/camera_info', self.on_cam_info, 1)

        # Kick off main sequence after sensors & Nav2 spin up
        self.create_timer(1.0, self.main_sequence, once=True)

        self.get_logger().info("CubeSeeker node initialized.")

    # Store latest color image
    def on_color(self, msg: Image):
        self.latest_color = msg

    # Store latest depth image
    def on_depth(self, msg: Image):
        self.latest_depth = msg

    # Store camera intrinsics once
    def on_cam_info(self, msg: CameraInfo):
        if self.cam_info is None:
            self.cam_info = msg

    def main_sequence(self):
        # 1) Record current position as home
        home_pose = self.nav.getCurrentPose()
        self.get_logger().info(
            f"Home recorded at ({home_pose.pose.position.x:.2f}, {home_pose.pose.position.y:.2f})"
        )

        # 2) Wait until Nav2 is ready
        self.get_logger().info("Waiting for Nav2 to become active...")
        self.nav.waitUntilNav2Active()

        # 3) Spin and scan for cube
        self.get_logger().info("Scanning for cube by rotating...")
        uv = self.scan_for_cube()
        if uv is None:
            self.get_logger().error("Cube not found – aborting.")
            rclpy.shutdown()
            return
        u, v = uv
        self.get_logger().info(f"Cube detected at pixel (u={u}, v={v})")

        # 4) Compute goal pose in map frame
        goal_pose = self.compute_map_goal(u, v)

        # 5) Navigate to cube and back home
        self.get_logger().info("Driving to cube...")
        self.nav.startToPose(goal_pose)
        self.get_logger().info("Returning to home pose...")
        self.nav.startToPose(home_pose)

        self.get_logger().info("Mission complete. Shutting down.")
        rclpy.shutdown()

    def scan_for_cube(self):
        """
        Rotate the robot and use find_uv() on each frame until it finds the cube.
        """
        pub = self.create_publisher(Twist, 'cmd_vel', 1)
        spin = Twist(); spin.angular.z = 0.4

        for _ in range(200):  # ~20 seconds at 10 Hz
            pub.publish(spin)
            rclpy.spin_once(self, timeout_sec=0.1)

            if None in (self.latest_color, self.latest_depth, self.cam_info):
                continue

            img = self.bridge.imgmsg_to_cv2(self.latest_color, 'bgr8')
            uv = self.find_uv(img)
            if uv:
                pub.publish(Twist())  # stop rotating
                return uv

        pub.publish(Twist())
        return None

    def find_uv(self, img: np.ndarray):
        """
        HSV threshold → contour → centroid of largest blob.
        """
        hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_min, self.hsv_max)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        c = max(cnts, key=cv2.contourArea)
        M = cv2.moments(c)
        if M['m00'] == 0:
            return None
        return int(M['m10']/M['m00']), int(M['m01']/M['m00'])

    def compute_map_goal(self, u: int, v: int):
        """
        Back-project (u,v) + depth → camera frame → map-frame PoseStamped.
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
