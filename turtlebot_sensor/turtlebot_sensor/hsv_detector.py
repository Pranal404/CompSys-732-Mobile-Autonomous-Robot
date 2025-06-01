#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool, Float32
from cv_bridge import CvBridge
import cv2
import numpy as np

class HSVDetector(Node):
    def __init__(self):
        super().__init__('hsv_detector')
        self.bridge = CvBridge()

        # ─── Parameters ──────────────────────────────────────────────────
        self.declare_parameter('image_topic',   'oakd/rgb/image_raw/compressed')
        self.declare_parameter('hsv_lower',     [20, 100, 100])
        self.declare_parameter('hsv_upper',     [40, 255, 255])
        self.declare_parameter('min_contour_area',    2000)
        self.declare_parameter('approx_epsilon_coeff',0.02)
        self.declare_parameter('focal_length_px',      600.0)
        self.declare_parameter('cube_size_mm',         30.0)
        self.declare_parameter('min_detect_dist_mm',   50.0)
        self.declare_parameter('max_detect_dist_mm',  300.0)

        # pull them out (use .value to get the actual Python list/int/float)
        topic = self.get_parameter('image_topic').value
        self.hsv_lower = np.array(self.get_parameter('hsv_lower').value, dtype=np.uint8)
        self.hsv_upper = np.array(self.get_parameter('hsv_upper').value, dtype=np.uint8)
        self.min_area = self.get_parameter('min_contour_area').value
        self.eps_coeff = self.get_parameter('approx_epsilon_coeff').value
        self.fpx      = self.get_parameter('focal_length_px').value
        self.L        = self.get_parameter('cube_size_mm').value
        self.min_d    = self.get_parameter('min_detect_dist_mm').value
        self.max_d    = self.get_parameter('max_detect_dist_mm').value

        # ─── OpenCV windows ───────────────────────────────────────────────
        cv2.namedWindow('Camera', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Mask',   cv2.WINDOW_NORMAL)

        # ─── ROS subscriptions & publishers ───────────────────────────────
        self.sub = self.create_subscription(
            CompressedImage, topic, self.cb_image, qos_profile_sensor_data
        )
        self.pub_detect = self.create_publisher(Bool,    'cube_detected',    1)
        self.pub_dist   = self.create_publisher(Float32, 'cube_distance_mm', 1)

        self.get_logger().info(f'🔍 HSV detector listening on `{topic}`')

    def cb_image(self, msg: CompressedImage):
        # decode compressed → BGR
        img = self.bridge.compressed_imgmsg_to_cv2(msg, 'bgr8')

        # 1) HSV threshold
        hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)

        # 2) clean mask
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)

        # 3) find & filter contours
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts or cv2.contourArea(max(cnts, key=cv2.contourArea)) < self.min_area:
            self._publish(False, -1.0); self._show(img, mask); return

        # 4) approximate to polygon
        c    = max(cnts, key=cv2.contourArea)
        peri = cv2.arcLength(c, True)
        pts  = cv2.approxPolyDP(c, self.eps_coeff * peri, True).reshape(-1,2)

        # prune hexagon → quad
        if len(pts) == 6:
            pts = self._prune_hexagon(pts)
        if len(pts) != 4:
            self._publish(False, -1.0); self._show(img, mask); return

        # 5) sort CCW around centroid
        ctr  = pts.mean(axis=0)
        angs = np.arctan2(pts[:,1]-ctr[1], pts[:,0]-ctr[0])
        rect = pts[np.argsort(angs)]

        # 6) compute distance
        lens   = [np.linalg.norm(rect[(i+1)%4] - rect[i]) for i in range(4)]
        mean_px = float(np.mean(lens))
        Z_mm    = (self.fpx * self.L) / mean_px
        ok      = (self.min_d <= Z_mm <= self.max_d)
        self._publish(ok, Z_mm)

        # 7) debug draw
        disp = img.copy()
        cv2.drawContours(disp, [rect.astype(int)], -1, (0,255,0), 2)
        for p in rect:
            cv2.circle(disp, tuple(p.astype(int)), 4, (0,0,255), -1)
        cv2.putText(disp, f"Z={Z_mm:.0f}mm", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
        self._show(disp, mask)

    def _prune_hexagon(self, pts):
        N, angles = len(pts), []
        for i, p in enumerate(pts):
            prev, nxt = pts[(i-1)%N]-p, pts[(i+1)%N]-p
            cosang = np.dot(prev,nxt)/(np.linalg.norm(prev)*np.linalg.norm(nxt))
            angles.append(np.degrees(np.arccos(np.clip(cosang, -1, 1))))
        return pts[np.argsort(angles)[:4]]

    def _publish(self, ok: bool, dist: float):
        self.pub_detect.publish(Bool(data=ok))
        self.pub_dist.publish(Float32(data=dist))

    def _show(self, img, mask):
        cv2.imshow('Camera', img)
        cv2.imshow('Mask',  mask)
        cv2.waitKey(1)

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = HSVDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
