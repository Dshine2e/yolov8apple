from glob import glob

from setuptools import find_packages, setup

package_name = 'asac_perception'
setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    tests_require=['pytest'],
    zip_safe=True,
    maintainer='Dshine2e',
    maintainer_email='dshine@users.noreply.github.com',
    description='D455 RGB-D YOLOv8 perception',
    license='Apache-2.0',
    entry_points={'console_scripts': [
        'detector = asac_perception.detector_node:main',
        'topview = asac_perception.topview_node:main',
        'train = asac_perception.train:main',
    ]},
)
