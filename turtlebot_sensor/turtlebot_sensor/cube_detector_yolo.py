#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
from cv_bridge import CvBridge
import cv2
import math
import numpy as np
from ultralytics import YOLO

class YoloCubeDetector(Node):
    def __init__(self):
        super().__init__('detect_cube')
        self.bridge = CvBridge()
        # parameters
        self.declare_parameter('rgb_topic',     'oakd/rgb/image_raw/compressed')
        self.declare_parameter('odom_topic',    'odom')
        self.declare_parameter('model_path',    '/afs/ec.auckland.ac.nz/users/p/i/ping440/unixhome/ros2_ws/src/turtlebot_sensor/models/best.pt')
        self.declare_parameter('fx',            1353.75)
        self.declare_parameter('cube_width',    0.08)
        self.declare_parameter('detect_topic',  'cube_detected')

        rgb_topic    = self.get_parameter('rgb_topic').get_parameter_value().string_value
        odom_topic   = self.get_parameter('odom_topic').get_parameter_value().string_value
        model_path   = self.get_parameter('model_path').get_parameter_value().string_value
        self.fx       = self.get_parameter('fx').get_parameter_value().double_value
        self.cube_w   = self.get_parameter('cube_width').get_parameter_value().double_value
        detect_topic = self.get_parameter('detect_topic').get_parameter_value().string_value

        self.model = YOLO(model_path)
        self.get_logger().info(f"✅ YOLO loaded: {model_path}")

        self.pose = None
        self.create_subscription(Odometry, odom_topic, self.odom_cb, 10)
        self.create_subscription(CompressedImage, rgb_topic,
                                 self.image_cb, qos_profile_sensor_data)

        self.pub = self.create_publisher(Bool, detect_topic, 10)

    def odom_cb(self, msg):
        self.pose = msg.pose.pose

    def image_cb(self, msg):
        if self.pose is None:
            return
        img = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        res = self.model(img, verbose=False)[0]
        found = False

        for box in res.boxes:
            if int(box.cls[0])==0 and float(box.conf[0])>0.5:
                x1,y1,x2,y2 = box.xyxy[0].cpu().numpy().astype(int)
                wpx = x2-x1
                Z   = (self.cube_w * self.fx) / wpx if wpx>0 else 0.0
                q = self.pose.orientation
                siny = 2*(q.w*q.z + q.x*q.y)
                cosy = 1-2*(q.y*q.y + q.z*q.z)
                yaw  = math.atan2(siny, cosy)
                # just global position in front of robot
                cx = self.pose.position.x + math.cos(yaw)*Z
                cy = self.pose.position.y + math.sin(yaw)*Z

                self.get_logger().info(f"🌍 Cube ≈ {Z:.2f}m ahead at ({cx:.2f},{cy:.2f})")
                self.pub.publish(Bool(data=True))
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(img, f"Z={Z:.2f}m",(x1,y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
                found=True
                break

        if not found:
            self.pub.publish(Bool(data=False))

        cv2.imshow("Cube Detector", img)
        cv2.waitKey(1)

def main():
    rclpy.init()
    node = YoloCubeDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__=='__main__':
    main()
