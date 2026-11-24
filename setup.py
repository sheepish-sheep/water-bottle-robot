from glob import glob

from setuptools import find_packages, setup

PKG = "water_bottle_robot"

setup(
    name=PKG,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + PKG]),
        ("share/" + PKG, ["package.xml"]),
        ("share/" + PKG + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    extras_require={"test": ["pytest"]},
    zip_safe=True,
    description="Stretch water-bottle robot",
    license="TODO: License declaration",
    entry_points={
        "console_scripts": [
            # manipulation: bottle handoff
            "handoff_sim = water_bottle_robot.manipulation.handoff_sim:main",
            "handoff_real = water_bottle_robot.manipulation.handoff_real:main",
            "handoff_real_check = water_bottle_robot.manipulation.handoff_real_check:main",
            "handoff_ros = water_bottle_robot.manipulation.handoff_ros:main",
            # navigation
            "go_to_fountain = water_bottle_robot.navigation.go_to_fountain:main",
            "obstacle_avoidance = water_bottle_robot.reactive.obstacle_avoidance:main",
            # perception
            "offered_bottle_detector = water_bottle_robot.perception.offered_bottle_detector:main",
            # sim test tools
            "nav_goal_test = water_bottle_robot.tools.nav_goal_test:main",
            "contact_replay = water_bottle_robot.tools.contact_replay:main",
        ],
    },
)
