from setuptools import setup
import os
from glob import glob

package_name = 'turtlebot_sensor'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
             ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
             glob('launch/*.launch.py')),
        # install the YOLO weights into share/<pkg>/models
        (os.path.join('share', package_name, 'models'),
            ['models/best.pt']),
    ],
    install_requires=[
        'setuptools',
        'ultralytics',            # for your YOLO detector
        'opencv-python-headless',  # if you need cv2
    ],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='you@example.com',
    description='Turtlebot sensor package',
    license='Apache License 2.0',
    entry_points={
        'console_scripts': [
            'turtle_explorer  = turtlebot_sensor.turtle_explorer:main',
            'turtle_path = turtlebot_sensor.turtle_path:main',
            'cube_detector_yolo = turtlebot_sensor.cube_detector_yolo:main',
            'hsv_detector = turtlebot_sensor.hsv_detector:main', 
            'return_node = turtlebot_sensor.return_node:main',
        ],
    },
)
