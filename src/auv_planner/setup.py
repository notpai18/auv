from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'auv_planner'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your_email@example.com',
    description='AUV Planner package for navigation and control',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # Gate mission
            'gate_navigator_node    = auv_planner.gate_navigator_node:main',
            'gate_navigator_v2_node = auv_planner.gate_navigator_v2_node:main',

            # Green zone mission
            'green_navigator_node   = auv_planner.green_navigator_node:main',

            # Depth control (thruster-based Z hold + cmd_vel merger)
            'depth_hold_node        = auv_planner.depth_hold_node:main',
            'cmd_vel_mixer_node     = auv_planner.cmd_vel_mixer_node:main',
        ],
    },
)