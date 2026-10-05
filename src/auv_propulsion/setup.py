from setuptools import find_packages, setup

package_name = 'auv_propulsion'

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
    description='AUV Propulsion package: 8-thruster allocation using pseudo-inverse TAM',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'thruster_allocator_node = auv_propulsion.thruster_allocator_node:main',
        ],
    },
)
