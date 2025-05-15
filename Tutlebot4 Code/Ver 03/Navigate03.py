#!/usr/bin/env python3
import rospy
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import actionlib
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from cv_bridge import CvBridge
import cv2
import numpy as np

class TurtleBotNavigator:
    def __init__(self):
        rospy.init_node('turtlebot_navigator')
        self.bridge = CvBridge()
        
        # Initialize action client for navigation
        self.move_base = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self.move_base.wait_for_server()
        
        # Subscribers
        self.image_sub = rospy.Subscriber('/camera/image_raw', Image, self.detect_cube)
        self.amcl_sub = rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.save_initial_pose)
        self.lidar_sub = rospy.Subscriber('/scan', LaserScan, self.avoid_obstacle)
        
        # Variables
        self.initial_pose = None
        self.cube_detected = False
        self.cube_position = None
        self.obstacle_detected = False

    def save_initial_pose(self, msg):
        """Save the initial position once at startup."""
        if not self.initial_pose:
            self.initial_pose = msg.pose.pose
            rospy.loginfo("Initial pose saved.")

    def detect_cube(self, img_msg):
        """Detect the Golden Cube using OpenCV."""
        if self.cube_detected:
            return
        
        cv_image = self.bridge.imgmsg_to_cv2(img_msg, 'bgr8')
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        
        # Define golden color range (adjust based on lighting)
        lower_gold = np.array([20, 100, 100])
        upper_gold = np.array([30, 255, 255])
        mask = cv2.inRange(hsv, lower_gold, upper_gold)
        
        # Detect contours and filter by size
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 1000:  # Adjust based on distance
                self.cube_detected = True
                self.cube_position = self.get_current_pose()
                rospy.loginfo(f"Golden Cube detected at {self.cube_position}")
                self.return_to_start()

    def avoid_obstacle(self, scan_msg):
        """Simple obstacle avoidance using LiDAR data."""
        ranges = np.array(scan_msg.ranges)
        min_distance = np.nanmin(ranges)
        if min_distance < 0.5:  # Threshold in meters
            self.obstacle_detected = True
            # Stop and rotate (replace with better logic)
            self.move_base.cancel_goal()
            goal = MoveBaseGoal()
            goal.target_pose.header.frame_id = 'base_link'
            goal.target_pose.pose.orientation.w = 1.0
            self.move_base.send_goal(goal)
        else:
            self.obstacle_detected = False

    def get_current_pose(self):
        """Get current robot pose from AMCL."""
        return rospy.wait_for_message('/amcl_pose', PoseWithCovarianceStamped).pose.pose

    def return_to_start(self):
        """Navigate back to the initial position."""
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = 'map'
        goal.target_pose.pose = self.initial_pose
        self.move_base.send_goal(goal)
        rospy.loginfo("Returning to start...")
        self.move_base.wait_for_result()
        rospy.loginfo("Successfully returned!")

    def explore(self):
        """Start exploration using explore_lite (assumes package is running)."""
        rospy.loginfo("Exploring the environment...")
        while not rospy.is_shutdown() and not self.cube_detected:
            rospy.sleep(0.1)

if __name__ == '__main__':
    navigator = TurtleBotNavigator()
    rospy.sleep(5)  # Wait for AMCL initialization
    navigator.explore()