#!/usr/bin/env python3
"""
Real-time Cube Seeker for TurtleBot4 with Nav2 (no map TF dependency)

This node:
 1. Waits for Nav2 to activate.
 2. Records the starting pose via the Navigator helper (odom frame).
 3. Optionally explores to build a map (SLAM optional).
 4. Moves forward manually via `/cmd_vel` until the cube is detected.
 5. Back-projects pixel+depth → goal in the odom frame.
 6. Uses Nav2 to navigate to the cube and return to start.
"""
import rclpy
from rclpy.node import Node
from turtlebot4_python_tutorials.nav_to_pose import TurtleBot4Navigator
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Twist, PoseStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
from rclpy.duration import Duration

class CubeSeeker(Node):
    def __init__(self):
        super().__init__('cube_seeker')
        # Nav2 helper
        self.nav = TurtleBot4Navigator()
        # OpenCV bridge
        self.bridge = CvBridge()
        # HSV thresholds for yellow cube
        self.hsv_min = np.array([15, 100, 100])
        self.hsv_max = np.array([45, 255, 255])
        # sensor data buffers
        self.latest_color = None
        self.latest_depth = None
        self.cam_info = None
        # cmd_vel publisher for manual movement
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        # camera subscriptions
        self.create_subscription(Image, '/camera/color/image_raw', self.on_color, 10)
        self.create_subscription(Image, '/camera/aligned_depth_to_color/image_raw', self.on_depth, 10)
        self.create_subscription(CameraInfo, '/camera/color/camera_info', self.on_cam_info, 10)
        # start sequence after delay
        self.start_timer = self.create_timer(1.0, self.main_sequence)
        self.get_logger().info('CubeSeeker initialized.')

    def on_color(self, msg: Image):
        self.latest_color = msg

    def on_depth(self, msg: Image):
        self.latest_depth = msg

    def on_cam_info(self, msg: CameraInfo):
        if self.cam_info is None:
            self.cam_info = msg

    def main_sequence(self):
        # cancel one-shot timer
        self.start_timer.cancel()

                # 1) Skip waiting for AMCL (basic_navigator)
        self.get_logger().info('Skipping AMCL wait; proceeding immediately.')

        # 2) Record home pose (odom frame) (odom frame)
        home = self.nav.getPoseStamped([0.0, 0.0], 0.0)
        self.get_logger().info(
            f'Home at x={home.pose.position.x:.2f}, y={home.pose.position.y:.2f}'
        )

        # 3) Explore (optional)
        self.get_logger().info('Exploring environment...')
        try:
            self.nav.explore()
            self.get_logger().info('Exploration complete.')
        except AttributeError:
            self.get_logger().info('No explore behavior; skipping.')

        # 4) Scan forward manually
        self.get_logger().info('Scanning for cube...')
        result = self.scan_forward_for_cube(step_size=0.3, max_steps=10, max_depth=5.0)
        if result is None:
            self.get_logger().error('Cube not found; aborting.')
            rclpy.shutdown()
            return
        u, v, depth_m = result
        self.get_logger().info(
            f'Cube detected at (u={u},v={v}), depth={depth_m:.2f} m'
        )

        # 5) Compute goal in odom frame
        goal = self.compute_goal_odom(u, v, depth_m)

        # 6) Navigate to cube and return
        self.get_logger().info('Driving to cube via Nav2...')
        self.nav.startToPose(goal)
        self.get_logger().info('Returning home via Nav2...')
        self.nav.startToPose(home)
        self.get_logger().info('Mission complete. Shutting down.')
        rclpy.shutdown()

    def scan_forward_for_cube(self, step_size, max_steps, max_depth):
        speed = 0.1  # m/s
        for i in range(max_steps):
            self.get_logger().info(f'Step {i+1}: moving forward {step_size} m')
            # move forward
            twist = Twist()
            twist.linear.x = speed
            duration = step_size / speed
            end_time = self.get_clock().now() + Duration(seconds=duration)
            while self.get_clock().now() < end_time:
                self.cmd_pub.publish(twist)
                rclpy.spin_once(self, timeout_sec=0.1)
            # stop
            self.cmd_pub.publish(Twist())
            rclpy.spin_once(self, timeout_sec=0.1)

            # check for cube
            if None in (self.latest_color, self.latest_depth, self.cam_info):
                continue
            img = self.bridge.imgmsg_to_cv2(self.latest_color, 'bgr8')
            uv = self.find_uv(img)
            if uv is None:
                continue
            u, v = uv
            depth_img = self.bridge.imgmsg_to_cv2(self.latest_depth, 'passthrough')
            d = float(depth_img[v, u]) * 0.001
            if 0.1 < d < max_depth:
                return u, v, d
            self.get_logger().info(f'Cube seen but out of range: {d:.2f} m')
        return None

    def find_uv(self, img: np.ndarray):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_min, self.hsv_max)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        c = max(cnts, key=cv2.contourArea)
        M = cv2.moments(c)
        if M['m00'] == 0:
            return None
        return int(M['m10']/M['m00']), int(M['m01']/M['m00'])

    def compute_goal_odom(self, u, v, depth_m):
        # back-project pixel into camera frame
        k = self.cam_info.k
        x = (u - k[2]) * depth_m / k[0]
        y = (v - k[5]) * depth_m / k[4]
        # transform into odom via Navigator helper
        return self.nav.transformPointToMap([x, y, depth_m])


def main():
    rclpy.init()
    node = CubeSeeker()
    rclpy.spin(node)
    node.get_logger().info('Shutting down rclpy')
    rclpy.shutdown()

if __name__ == '__main__':
    main()
