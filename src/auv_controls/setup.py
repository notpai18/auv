from setuptools import find_packages, setup

package_name = 'auv_controls'

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
    maintainer='pai',
    maintainer_email='pranav2pai@gmail.com',
    description='AUV Controls package: closed-loop 6-DOF velocity and attitude controller',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'velocity_controller_node = auv_controls.velocity_controller_node:main',
        ],
    },
)
