import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'auv_sensors'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='pai',
    maintainer_email='pranav2pai@gmail.com',
    description='Sensor layer for AUV: Depth, Leak, Battery, IMU, and Magnetometer nodes',
    license='MIT',
    entry_points={
        'console_scripts': [
            'depth_sensor = auv_sensors.depth_sensor:main',
            'depth_sensor.py = auv_sensors.depth_sensor:main',
            'leak_mock = auv_sensors.leak_mock:main',
            'leak_mock.py = auv_sensors.leak_mock:main',
            'battery_mock = auv_sensors.battery_mock:main',
            'battery_mock.py = auv_sensors.battery_mock:main',
            'imu_mock = auv_sensors.imu_mock:main',
            'imu_mock.py = auv_sensors.imu_mock:main',
            'dvl_sensor = auv_sensors.dvl_sensor:main',
            'dvl_sensor.py = auv_sensors.dvl_sensor:main',
            'sensor_fusion_node = auv_sensors.sensor_fusion_node:main',
            'sensor_fusion_node.py = auv_sensors.sensor_fusion_node:main',
            'safety_supervisor_node = auv_sensors.safety_supervisor_node:main',
            'safety_supervisor_node.py = auv_sensors.safety_supervisor_node:main',
        ],
    },
)
