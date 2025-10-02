from setuptools import setup
from glob import glob

package_name = 'my_vslam_bringup'

setup(
    name=package_name,
    version='0.0.1',
    packages=[],  # no python modules; just a launch file
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='',
    maintainer_email='',
    description='Bringup for Isaac ROS Visual SLAM',
    license='',
    tests_require=['pytest'],
    entry_points={'console_scripts': []},
)
