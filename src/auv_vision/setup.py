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
            # Gate detection pipeline
            'gate_detector_node  = auv_vision.gate_detector_node:main',
            'gate_localizer_node = auv_vision.gate_localizer_node:main',

            # Green zone detection pipeline
            'green_detector_node  = auv_vision.green_detector_node:main',
            'green_localizer_node = auv_vision.green_localizer_node:main',

            # Blue bin detection pipeline
            'blue_bin_detector_node    = auv_vision.blue_bin_detector_node:main',
            'blue_bin_cv_detector_node = auv_vision.blue_bin_cv_detector_node:main',
            'blue_bin_localizer_node   = auv_vision.blue_bin_localizer_node:main',

            # Display-only: Beer-Lambert underwater haze on the left camera.
            # Publishes a separate topic; detector inputs are untouched.
            'underwater_view_node = auv_vision.underwater_view_node:main',
        ],
    },
)