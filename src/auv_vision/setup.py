from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'auv_vision'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # This line ensures your weights file is installed and accessible at runtime
        (os.path.join('share', package_name, 'weights'), glob('weights/*.pt')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your_email@example.com',
    description='AUV Vision package for object detection using YOLOv8',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # This creates the executable command: ros2 run auv_vision gate_detector_node
            'gate_detector_node = auv_vision.gate_detector_node:main'
        ],
    },
)