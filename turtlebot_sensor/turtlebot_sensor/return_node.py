#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry, OccupancyGrid, Path
from std_msgs.msg import Bool
import numpy as np
import math
from queue import PriorityQueue
from scipy.ndimage import distance_transform_edt

class ReturnNode(Node):
    def __init__(self):
        super().__init__('return_node')
        self.map_data = None
        self.map_resolution = None
        self.map_origin = None
        self.robot_pos = None
        self.robot_yaw = 0.0
        self.start_pos = None
        self.path = []
        self.triggered = False

        self.create_subscription(OccupancyGrid, 'map', self.map_callback, 10)
        self.create_subscription(Odometry, 'odom', self.odom_callback, 10)
        self.create_subscription(Bool, 'cube_detected', self.cube_callback, 10)

        self.cmd_pub  = self.create_publisher(Twist, 'cmd_vel', 10)
        self.path_pub = self.create_publisher(Path, 'return_path', 10)

        self.timer = self.create_timer(0.2, self.follow_path)

    def map_callback(self, msg):
        self.map_data = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        self.map_resolution = msg.info.resolution
        self.map_origin = (msg.info.origin.position.x, msg.info.origin.position.y)

    def odom_callback(self, msg):
        self.robot_pos = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        if self.start_pos is None:
            self.start_pos = self.robot_pos
            self.get_logger().info(f"🏠 Start saved at {self.start_pos}")
        q = msg.pose.pose.orientation
        siny = 2*(q.w*q.z + q.x*q.y)
        cosy = 1-2*(q.y*q.y + q.z*q.z)
        self.robot_yaw = math.atan2(siny, cosy)

    def cube_callback(self, msg):
        if msg.data and not self.triggered:
            self.triggered = True
            self.get_logger().info("🎯 Cube detected → Planning return path")
            if self.map_data is None or self.robot_pos is None:
                self.get_logger().error("⚠️ Cannot plan return: no map or position")
                return
            start = self.world_to_grid(*self.robot_pos)
            goal  = self.world_to_grid(*self.start_pos)
            self.path = self.astar(start, goal)
            self.publish_path(self.path)

    def publish_path(self, path):
        msg = Path()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        for gx, gy in path:
            p = PoseStamped()
            p.header = msg.header
            p.pose.position.x, p.pose.position.y = self.grid_to_world(gx, gy)
            p.pose.orientation.w = 1.0
            msg.poses.append(p)
        self.path_pub.publish(msg)

    def follow_path(self):
        if not self.triggered or not self.path or self.robot_pos is None:
            return
        gx, gy = self.path[0]
        wx, wy = self.grid_to_world(gx, gy)
        dx, dy = wx - self.robot_pos[0], wy - self.robot_pos[1]
        dist = math.hypot(dx, dy)
        ang  = math.atan2(dy, dx)
        err  = math.atan2(math.sin(ang - self.robot_yaw), math.cos(ang - self.robot_yaw))
        if dist < 0.1:
            self.path.pop(0)
            if not self.path:
                self.cmd_pub.publish(Twist())
                self.get_logger().info("✅ Arrived back at home.")
            return
        cmd = Twist()
        cmd.linear.x  = max(0.15, min(0.4, 0.5 * dist))
        cmd.angular.z = max(-1.0, min(1.0, 1.5 * err))
        self.cmd_pub.publish(cmd)

    def world_to_grid(self, x, y):
        return (int((x - self.map_origin[0]) / self.map_resolution),
                int((y - self.map_origin[1]) / self.map_resolution))

    def grid_to_world(self, gx, gy):
        return (self.map_origin[0] + gx * self.map_resolution,
                self.map_origin[1] + gy * self.map_resolution)

    def astar(self, start, goal):
        h, w = self.map_data.shape
        queue = PriorityQueue()
        queue.put((0, start))
        came_from = {}
        g_score = {start: 0}
        dirs = [(-1,0),(1,0),(0,-1),(0,1)]
        obst = np.logical_or(self.map_data == 100, self.map_data == -1)
        inflate = distance_transform_edt(~obst) * self.map_resolution

        while not queue.empty():
            _, current = queue.get()
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                return list(reversed(path))

            for dx, dy in dirs:
                nb = (current[0] + dx, current[1] + dy)
                if not (0 <= nb[0] < w and 0 <= nb[1] < h):
                    continue
                if self.map_data[nb[1], nb[0]] != 0:
                    continue
                if nb != goal and inflate[nb[1], nb[0]] < 0.15:
                    continue
                tg = g_score[current] + 1
                if nb not in g_score or tg < g_score[nb]:
                    g_score[nb] = tg
                    came_from[nb] = current
                    f = tg + math.hypot(goal[0] - nb[0], goal[1] - nb[1])
                    queue.put((f, nb))
        return []

def main():
    rclpy.init()
    node = ReturnNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
