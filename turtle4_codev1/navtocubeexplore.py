#!/usr/bin/env python3
"""
Real-time Cube Seeker for TurtleBot4

This node:
 1. Records start pose via Navigator (odom frame).
 2. Optionally explores to build a map.
 3. Scans environment with LiDAR to avoid obstacles during manual forward steps.
 4. Manually steps forward via `cmd_vel`, avoiding obstacles, until cube detected.
 5. Back-projects pixel+depth → goal in odom frame.
 6. Commands Nav2 to drive to cube and return.
"""
import rclpy
from rclpy.node import Node
from turtlebot4_python_tutorials.nav_to_pose import TurtleBot4Navigator
from sensor_msgs.msg import Image, CameraInfo, LaserScan
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
        # HSV thresholds for golden cube detection
        self.hsv_min = np.array([20, 80, 80])
        self.hsv_max = np.array([40, 255, 255])
        # Sensor buffers
        self.latest_color = None
        self.latest_depth = None
        self.cam_info = None
        self.latest_scan = None
        # Publishers/subscribers on relative topics (namespace auto-applied)
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.create_subscription(LaserScan, 'scan', self.on_scan, 10)
        self.create_subscription(Image, 'camera/color/image_raw', self.on_color, 10)
        self.create_subscription(Image, 'camera/aligned_depth_to_color/image_raw', self.on_depth, 10)
        self.create_subscription(CameraInfo, 'camera/color/camera_info', self.on_cam_info, 10)
        # Start main sequence after short delay
        self.create_timer(1.0, self.main_sequence)
        self.get_logger().info('CubeSeeker initialized on relative topics.')

    def on_color(self, msg: Image):
        self.latest_color = msg

    def on_depth(self, msg: Image):
        self.latest_depth = msg

    def on_cam_info(self, msg: CameraInfo):
        if self.cam_info is None:
            self.cam_info = msg

    def on_scan(self, msg: LaserScan):
        self.latest_scan = msg

    def main_sequence(self):
        # record start pose
        home = self.nav.getPoseStamped([0.0, 0.0], 0.0)
        self.get_logger().info(
            f'Home at x={home.pose.position.x:.2f}, y={home.pose.position.y:.2f}'
        )
        # optional exploration
        try:
            self.nav.explore()
            self.get_logger().info('Exploration complete.')
        except AttributeError:
            self.get_logger().info('No explore behavior; skipping.')
        # scan forward
        self.get_logger().info('Scanning for cube with obstacle avoidance...')
        result = self.scan_forward_for_cube(step_size=0.3, max_steps=20, max_depth=5.0)
        if result is None:
            self.get_logger().error('Cube not found; aborting.')
            rclpy.shutdown()
            return
        u, v, depth_m = result
        self.get_logger().info(
            f'Cube detected at (u={u},v={v}), depth={depth_m:.2f} m'
        )
        # compute goal and navigate
        goal = self.compute_goal_odom(u, v, depth_m)
        self.get_logger().info('Driving to cube via Nav2...')
        self.nav.startToPose(goal)
        self.get_logger().info('Returning home via Nav2...')
        self.nav.startToPose(home)
        self.get_logger().info('Mission complete. Shutting down.')
        rclpy.shutdown()

    def scan_forward_for_cube(self, step_size, max_steps, max_depth):
        speed = 0.1  # m/s
        avoid_dist = 0.5  # meters
        for i in range(max_steps):
            self.get_logger().info(f'Step {i+1}: forward {step_size} m')
            # obstacle check
            if self.latest_scan is not None:
                ranges = np.array(self.latest_scan.ranges)
                mid = len(ranges) // 2
                front = ranges[mid-10:mid+10]
                front = front[np.isfinite(front)]
                if front.size and front.min() < avoid_dist:
                    self.get_logger().warn('Obstacle ahead; rotating to avoid')
                    self.rotate_in_place(0.5)
                    continue
            # move forward
            twist = Twist()
            twist.linear.x = speed
            end_time = self.get_clock().now() + Duration(seconds=step_size / speed)
            while self.get_clock().now() < end_time:
                self.cmd_pub.publish(twist)
                rclpy.spin_once(self, timeout_sec=0.1)
            # stop
            self.cmd_pub.publish(Twist())
            rclpy.spin_once(self, timeout_sec=0.1)
            # sensor check
            if self.latest_color is None or self.latest_depth is None or self.cam_info is None:
                self.get_logger().warn('Waiting for sensor data; retrying step.')
                continue
            # detect cube
            img = self.bridge.imgmsg_to_cv2(self.latest_color, 'bgr8')
            uv = self.find_uv(img)
            if uv is None:
                self.get_logger().info('No cube contours this step.')
                continue
            u, v = uv
            depth_img = self.bridge.imgmsg_to_cv2(self.latest_depth, 'passthrough')
            d = float(depth_img[v, u]) * 0.001
            self.get_logger().info(f'Depth at cube: {d:.2f} m')
            if 0.1 < d < max_depth:
                return u, v, d
            self.get_logger().info(f'Cube out of range: {d:.2f} m')
        return None

    def rotate_in_place(self, angular_speed):
        twist = Twist()
        twist.angular.z = angular_speed
        end = self.get_clock().now() + Duration(seconds=1.0)
        while self.get_clock().now() < end:
            self.cmd_pub.publish(twist)
            rclpy.spin_once(self, timeout_sec=0.1)
        self.cmd_pub.publish(Twist())
        rclpy.spin_once(self, timeout_sec=0.1)

    def find_uv(self, img: np.ndarray):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_min, self.hsv_max)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        filtered = [c for c in cnts if cv2.contourArea(c) > 2000]
        if not filtered:
            return None
        c = max(filtered, key=cv2.contourArea)
        M = cv2.moments(c)
        if M['m00'] == 0:
            return None
        u = int(M['m10'] / M['m00'])
        v = int(M['m01'] / M['m00'])
        return u, v

    def compute_goal_odom(self, u, v, depth_m):
        k = self.cam_info.k
        x = (u - k[2]) * depth_m / k[0]
        y = (v - k[5]) * depth_m / k[4]
        return self.nav.transformPointToMap([x, y, depth_m])


def main():
    rclpy.init()
    node = CubeSeeker()
    rclpy.spin(node)
    node.get_logger().info('Shutting down rclpy')
    rclpy.shutdown()

if __name__ == '__main__':
    main()
