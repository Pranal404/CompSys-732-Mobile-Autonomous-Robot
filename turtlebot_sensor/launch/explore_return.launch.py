#!/usr/bin/env python3
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    robot_ns = LaunchConfiguration('robot_ns')
    return LaunchDescription([
        DeclareLaunchArgument(
            'robot_ns',
            default_value='',
            description='Namespace for all nodes'
        ),
  
        Node(
            package='turtlebot_sensor',
            executable='turtle_explorer',
            name='explorer_node',
            namespace=robot_ns,
            output='screen'
        ),
        Node(
            package='turtlebot_sensor',
            executable='turtle_path',
            name='path_follower_node',
            namespace=robot_ns,
            output='screen'
        ),
        Node(
            package='turtlebot_sensor',
            executable='cube_detector_yolo',
            name='detect_cube',
            namespace=robot_ns,
            output='screen',
            parameters=[{
                'rgb_topic':     'oakd/rgb/image_raw/compressed',
                'odom_topic':    'odom',
                'model_path':    '/afs/ec.auckland.ac.nz/users/p/i/ping440/unixhome/ros2_ws/src/turtlebot_sensor/models/best.pt',
                'fx':            1353.75,
                'cube_width':    0.08,
                'detect_topic':  'cube_detected'
            }]
        ),
    ])
"""
#!/usr/bin/env python3
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition

def generate_launch_description():
    detector = LaunchConfiguration('detector', default='yolo')

    return LaunchDescription([
        DeclareLaunchArgument(
            'detector',
            default_value='yolo',
            description='Detection method: "yolo" or "hsv"'
        ),

        # ─── Explorer Node ─────────────────────────────
        Node(
            package='turtlebot_sensor',
            executable='explorer_node',
            name='explorer_node',
            namespace='T17',
            output='screen'
        ),

        # ─── Cube Detector (YOLO or HSV) ──────────────
        Node(
            condition=IfCondition(detector.perform(None) == 'yolo'),
            package='turtlebot_sensor',
            executable='detect_cube',
            name='detect_cube',
            namespace='T17',
            output='screen',
            parameters=[{
                'rgb_topic':    'camera/rgb/image_raw/compressed',
                'odom_topic':   'odom',
                'model_path':   '/absolute/path/to/best.pt',
                'fx':           1353.75,
                'cube_width':   0.08,
                'detect_topic': 'cube_detected'
            }]
        ),

        Node(
            condition=IfCondition(detector.perform(None) == 'hsv'),
            package='turtlebot_sensor',
            executable='hsv_detector',
            name='hsv_detector',
            namespace='T17',
            output='screen',
            parameters=[{
                'image_topic':  'camera/rgb/image_raw/compressed',
                'cube_detected': 'cube_detected'
            }]
        ),

        # ─── Return Node (A* navigation) ───────────────
        Node(
            package='turtlebot_sensor',
            executable='return_node',
            name='return_node',
            namespace='T17',
            output='screen'
        ),
    ])
