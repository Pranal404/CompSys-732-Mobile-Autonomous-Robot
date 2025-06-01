#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PointStamped, PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from std_msgs.msg import Bool
from visualization_msgs.msg import Marker
import numpy as np
import math
from queue import PriorityQueue
from scipy.ndimage import distance_transform_edt

class PathFollowerNode(Node):
    def __init__(self):
        super().__init__('path_follower_node')
        self.map_data = None;   self.map_resolution = None;   self.map_origin = None
        self.robot_pos = None;  self.robot_yaw = 0.0;         self.start_pos = None
        self.cube_found = False
        self.spinning = True
        self.spin_start = self.get_clock().now()
        self.spin_dur = 6.0
        self.last_pos = None
        self.last_move = self.get_clock().now()
        self.recovery = False
        self.recovery_start = None
        self.cooldown_until = self.get_clock().now()

        # subs & pubs
        self.create_subscription(OccupancyGrid, 'map', self.map_callback, 10)
        self.create_subscription(Odometry, 'odom', self.pose_callback, 10)
        self.create_subscription(PointStamped, 'target_point', self.goal_callback, 10)
        self.create_subscription(Bool, 'cube_detected', self.cube_callback, 10)

        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.path_pub = self.create_publisher(Path, 'a_star_path', 10)
        self.goal_marker_pub = self.create_publisher(Marker, 'goal_marker', 10)

        self.path = []
        self.timer = self.create_timer(0.2, self.follow_path)

    def map_callback(self, msg):
        self.map_data = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        self.map_resolution = msg.info.resolution
        self.map_origin = (msg.info.origin.position.x, msg.info.origin.position.y)

    def pose_callback(self, msg):
        self.robot_pos = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        if self.start_pos is None:
            self.start_pos = self.robot_pos
            self.get_logger().info(f"🏠 Start saved at {self.start_pos}")
        q = msg.pose.pose.orientation
        siny = 2*(q.w*q.z + q.x*q.y)
        cosy = 1-2*(q.y*q.y + q.z*q.z)
        self.robot_yaw = math.atan2(siny, cosy)
        if self.last_pos is not None:
            d = math.hypot(self.robot_pos[0]-self.last_pos[0], self.robot_pos[1]-self.last_pos[1])
            if d>0.005:
                self.last_move = self.get_clock().now()
        self.last_pos = self.robot_pos

    def cube_callback(self, msg):
        if msg.data and not self.cube_found:
            self.cube_found = True
            self.get_logger().info("🎯 Cube! plotting return path.")
            start = self.world_to_grid(*self.robot_pos)
            goal  = self.world_to_grid(*self.start_pos)
            self.path = self.astar(start, goal)
            self.publish_path(self.path)

    def goal_callback(self, msg):
        if self.cube_found or self.map_data is None or self.robot_pos is None:
            return
        s = self.world_to_grid(*self.robot_pos)
        g = self.world_to_grid(msg.point.x, msg.point.y)
        if self.map_data[s[1],s[0]]!=0 or self.map_data[g[1],g[0]]==100:
            return
        self.path = self.astar(s, g)
        self.publish_path(self.path)
        self.publish_goal_marker((msg.point.x, msg.point.y))

    def publish_goal_marker(self, gw):
        m=Marker()
        m.header.frame_id="map";m.ns="goal";m.id=0;m.type=m.SPHERE;m.action=m.ADD
        m.pose.position.x, m.pose.position.y = gw; m.pose.position.z=0.1
        m.scale.x=m.scale.y=m.scale.z=0.2
        m.color.r=1.0; m.color.g=0.2; m.color.b=0.2; m.color.a=1.0
        self.goal_marker_pub.publish(m)

    def publish_path(self, path):
        msg=Path(); msg.header.frame_id="map"; msg.header.stamp=self.get_clock().now().to_msg()
        for gx,gy in path:
            ps=PoseStamped(); ps.header=msg.header
            ps.pose.position.x, ps.pose.position.y = self.grid_to_world(gx,gy)
            ps.pose.orientation.w=1.0
            msg.poses.append(ps)
        self.path_pub.publish(msg)

    def follow_path(self):
        now = self.get_clock().now()
        # initial spin
        if self.spinning:
            if (now-self.spin_start).nanoseconds/1e9 < self.spin_dur:
                t=Twist(); t.angular.z = 2*math.pi/self.spin_dur
                self.cmd_pub.publish(t); return
            else:
                self.spinning=False
                self.cmd_pub.publish(Twist()); return
        # recovery spin
        if self.recovery:
            if (now-self.recovery_start).nanoseconds/1e9<3.0:
                t=Twist(); t.angular.z=math.pi/1.2
                self.cmd_pub.publish(t); return
            else:
                self.recovery=False
                self.last_move = now
                self.cmd_pub.publish(Twist()); return
        # stuck?
        if (now-self.last_move).nanoseconds/1e9>10.0 and not self.recovery and now>self.cooldown_until:
            self.recovery=True
            self.recovery_start = now
            self.cooldown_until = now + rclpy.duration.Duration(seconds=5)
            return
        # follow path
        if not self.path or self.robot_pos is None:
            return
        gx,gy = self.path[0]
        wx,wy = self.grid_to_world(gx,gy)
        dx,dy = wx-self.robot_pos[0], wy-self.robot_pos[1]
        dist = math.hypot(dx,dy)
        ang  = math.atan2(dy,dx)
        err  = math.atan2(math.sin(ang-self.robot_yaw), math.cos(ang-self.robot_yaw))
        if dist<0.1:
            self.path.pop(0)
            if not self.path and self.cube_found:
                self.cmd_pub.publish(Twist())
                self.get_logger().info("🏁 Returned home.")
            return
        t=Twist()
        t.linear.x  = max(0.15, min(0.4,0.5*dist))
        t.angular.z = max(-1.0, min(1.0,0.5*err))
        self.cmd_pub.publish(t)

    def world_to_grid(self,x,y):
        return (int((x-self.map_origin[0])/self.map_resolution),
                int((y-self.map_origin[1])/self.map_resolution))

    def grid_to_world(self,gx,gy):
        return (self.map_origin[0]+gx*self.map_resolution,
                self.map_origin[1]+gy*self.map_resolution)

    def astar(self,start,goal):
        h,w = self.map_data.shape
        open_set = PriorityQueue(); open_set.put((0,start))
        came_from={}; g_score={start:0}
        dirs=[(-1,0),(1,0),(0,-1),(0,1)]
        obst = np.logical_or(self.map_data==100, self.map_data==-1)
        infl = distance_transform_edt(~obst)*self.map_resolution
        while not open_set.empty():
            _,cur = open_set.get()
            if cur==goal:
                path=[]; c=cur
                while c in came_from:
                    path.append(c); c=came_from[c]
                return list(reversed(path))
            for dx,dy in dirs:
                nb=(cur[0]+dx, cur[1]+dy)
                if not (0<=nb[0]<w and 0<=nb[1]<h): continue
                if self.map_data[nb[1],nb[0]]!=0: continue
                if nb!=goal and infl[nb[1],nb[0]]<0.15: continue
                tg = g_score[cur]+1
                if nb not in g_score or tg<g_score[nb]:
                    g_score[nb]=tg
                    came_from[nb]=cur
                    f = tg + math.hypot(goal[0]-nb[0], goal[1]-nb[1])
                    open_set.put((f,nb))
        return []

def main():
    rclpy.init()
    node = PathFollowerNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__=='__main__':
    main()
