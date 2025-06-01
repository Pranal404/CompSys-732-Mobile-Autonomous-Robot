#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker
from std_msgs.msg import Bool
import numpy as np
import math
from scipy.ndimage import distance_transform_edt, binary_dilation

class SweepExplorerNode(Node):
    def __init__(self):
        super().__init__('explorer_node')
        self.map = None
        self.pose = None
        self.yaw = 0.0
        self.map_resolution = None
        self.map_origin = None
        self.visited_map = None
        self.last_goal = None
        self.last_goal_time = 0.0
        self.last_position = None
        self.exploring = True

        # subscriptions & publishers
        self.create_subscription(OccupancyGrid, 'map', self.map_callback, 10)
        self.create_subscription(Odometry, 'odom', self.odom_callback, 10)
        self.create_subscription(Bool, 'cube_detected', self.cube_callback, 10)
        self.goal_pub   = self.create_publisher(PointStamped, 'target_point', 10)
        self.marker_pub = self.create_publisher(Marker, 'unsafe_zone_marker', 10)

        # 1 Hz frontier‐selection
        self.timer = self.create_timer(1.0, self.publish_sweep_goal)

    def map_callback(self, msg):
        self.map = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        self.map_resolution = msg.info.resolution
        self.map_origin = (msg.info.origin.position.x, msg.info.origin.position.y)
        if self.visited_map is None or self.visited_map.shape != self.map.shape:
            self.visited_map = np.zeros_like(self.map, dtype=bool)

    def odom_callback(self, msg):
        self.pose = msg.pose.pose.position
        q = msg.pose.pose.orientation
        siny = 2*(q.w*q.z + q.x*q.y)
        cosy = 1 - 2*(q.y*q.y + q.z*q.z)
        self.yaw = math.atan2(siny, cosy)

        if self.map is not None:
            gx = int((self.pose.x - self.map_origin[0]) / self.map_resolution)
            gy = int((self.pose.y - self.map_origin[1]) / self.map_resolution)
            if 0<=gx<self.map.shape[1] and 0<=gy<self.map.shape[0]:
                self.visited_map[gy, gx] = True

    def cube_callback(self, msg):
        if msg.data and self.exploring:
            self.get_logger().info("📦 Cube detected → stopping exploration")
            self.exploring = False
            self.timer.cancel()

    def publish_sweep_goal(self):
        if not self.exploring or self.map is None or self.pose is None:
            return

        now = self.get_clock().now().nanoseconds / 1e9
        if self.last_goal and math.hypot(self.pose.x - self.last_position[0],
                                         self.pose.y - self.last_position[1])<0.1 and \
           now - self.last_goal_time < 3.0:
            return  # wait for robot to move

        unexplored = (self.map==0) & (~self.visited_map)
        occupied   = (self.map==100)
        unknown    = (self.map==-1)

        inflate_occ = binary_dilation(occupied, iterations=round(0.25/self.map_resolution))
        inflate_unk = binary_dilation(unknown, iterations=round(0.25/self.map_resolution))
        unsafe      = inflate_occ | inflate_unk
        safe_mask   = ~unsafe

        # visualize
        m = Marker()
        m.header.frame_id = "map"; m.ns="unsafe"; m.type=m.CUBE_LIST; m.action=m.ADD
        m.scale.x=m.scale.y=self.map_resolution; m.scale.z=0.01
        m.color.a=0.5; m.color.r=1.0
        for y,x in np.argwhere(unsafe):
            p = PointStamped().point
            p.x = self.map_origin[0] + x*self.map_resolution
            p.y = self.map_origin[1] + y*self.map_resolution
            p.z = 0.0
            m.points.append(p)
        self.marker_pub.publish(m)

        explorable = unexplored & safe_mask
        idxs = np.argwhere(explorable)
        if idxs.size==0:
            self.get_logger().info("✅ Exploration done.")
            return

        # distance‐transform clearance
        obst_mask = np.logical_or(occupied, unknown)
        inflated  = distance_transform_edt(~obst_mask)*self.map_resolution
        cleared   = distance_transform_edt(~obst_mask)

        best_score, best_pt = -1, None
        for fy,fx in idxs:
            wx = self.map_origin[0] + fx*self.map_resolution
            wy = self.map_origin[1] + fy*self.map_resolution
            dx,dy = wx-self.pose.x, wy-self.pose.y
            ang = abs(self.normalize_angle(math.atan2(dy,dx)-self.yaw))
            dist = math.hypot(dx,dy)
            buf   = cleared[fy,fx]*self.map_resolution
            if inflated[fy,fx]<0.15 or ang>math.pi/3 or dist<1.5 or buf<0.35:
                continue
            score = dist + 4*buf - 3*ang
            if score>best_score:
                best_score, best_pt = score, (wx,wy)

        if best_pt:
            goal = PointStamped()
            goal.header.frame_id="map"
            goal.point.x, goal.point.y = best_pt
            self.goal_pub.publish(goal)
            self.last_goal = best_pt
            self.last_goal_time = now
            self.last_position = (self.pose.x, self.pose.y)
            self.get_logger().info(f"🧭 New frontier: {best_pt}")
        else:
            self.get_logger().warn("❌ No valid frontier")

    @staticmethod
    def normalize_angle(a):
        while a>math.pi: a-=2*math.pi
        while a<-math.pi: a+=2*math.pi
        return a

def main():
    rclpy.init()
    node = SweepExplorerNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__=='__main__':
    main()
