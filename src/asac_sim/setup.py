from glob import glob
from setuptools import find_packages, setup

setup(
    name="asac_sim",
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/asac_sim"]),
        ("share/asac_sim", ["package.xml"]),
        ("share/asac_sim/launch", glob("launch/*.launch.py")),
        ("share/asac_sim/config", glob("config/*")),
        ("share/asac_sim/assets/piper", glob("assets/piper/*.*") + ["assets/piper/LICENSE"]),
        ("share/asac_sim/assets/piper/meshes", glob("assets/piper/meshes/*.STL")),
    ],
    install_requires=["setuptools"],
    tests_require=["pytest"],
    zip_safe=True,
    maintainer="Dshine2e",
    maintainer_email="dshine@users.noreply.github.com",
    description="Rendered fixed top camera simulation",
    license="Apache-2.0 AND MIT",
    entry_points={
        "console_scripts": [
            "scene = asac_sim.sim_node:main",
            "evaluate = asac_sim.evaluate_node:main",
        ]
    },
)
