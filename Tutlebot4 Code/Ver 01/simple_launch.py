from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(package='turtlebot_navigation',
             executable='simple_autonomy',
             name='simple_autonomy'),
    ])
