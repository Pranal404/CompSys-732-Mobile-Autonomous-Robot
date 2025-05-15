#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
import tf2_ros, tf2_geometry_msgs
import time

class SimpleAutonomy(Node):
    def __init__(self):
        super().__init__('simple_autonomy')
        # high-level navigator
        self.navigator = BasicNavigator()
        # camera → cv image
        self.bridge = CvBridge()
        self.cube_pose_map = None

        # subs / pubs
        self.create_subscription(Image, '/camera/color/image_raw', self.on_image, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # TF to convert camera→map
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

    def on_image(self, msg: Image):
        if self.cube_pose_map:
            return  # already found
        img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (20,100,100), (40,255,255))
        cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: return
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) < 5000: return

        x,y,w,h = cv2.boundingRect(c)
        cx, cy = x + w/2, y + h/2
        depth = 1.0  # assume 1 m ahead

        ps = PoseStamped()
        ps.header.stamp    = self.get_clock().now().to_msg()
        ps.header.frame_id = 'camera_link'
        ps.pose.position.x = depth
        ps.pose.position.y = (cx - img.shape[1]/2) * 0.0015
        ps.pose.position.z = (cy - img.shape[0]/2) * 0.0015
        ps.pose.orientation.w = 1.0

        # transform into map frame
        try:
            tf = self.tf_buffer.lookup_transform('map', 'camera_link', rclpy.time.Time())
            self.cube_pose_map = tf2_geometry_msgs.do_transform_pose(ps, tf)
            self.get_logger().info('Cube detected in map frame, ready to approach')
        except Exception as e:
            self.get_logger().warn(f'TF failed: {e}')

    def run(self):
        # wait for Nav2
        while not self.navigator.server_up():
            self.get_logger().info('Waiting for Nav2...')
            time.sleep(0.5)

        # record start
        start = self.navigator.get_current_pose().pose
        self.get_logger().info(f'Start pose: {start.position}')

        # wander until detected
        twist = Twist()
        rate = self.create_rate(10)
        self.get_logger().info('Wandering...')
        while rclpy.ok() and not self.cube_pose_map:
            twist.linear.x  = 0.2
            twist.angular.z = 0.2
            self.cmd_pub.publish(twist)
            rclpy.spin_once(self, timeout_sec=0.1)
        twist.linear.x = twist.angular.z = 0.0
        self.cmd_pub.publish(twist)

        # approach
        if self.cube_pose_map:
            self.get_logger().info('Approaching cube…')
            res = self.navigator.goToPose(self.cube_pose_map)
            if res == TaskResult.SUCCEEDED:
                self.get_logger().info('Reached cube!')
            else:
                self.get_logger().warn('Could not reach cube')

        # return home
        self.get_logger().info('Returning to start…')
        res2 = self.navigator.goToPose(start)
        if res2 == TaskResult.SUCCEEDED:
            self.get_logger().info('Home sweet home!')
        else:
            self.get_logger().warn('Return failed')

def main(args=None):
    rclpy.init(args=args)
    node = SimpleAutonomy()
    node.run()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
