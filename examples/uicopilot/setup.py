from setuptools import find_packages, setup

setup(
    name="uicopilot",
    version="0.1.0",
    description="Unofficial pip-installable version of UICopilot",
    packages=find_packages("src"),
    package_dir={"": "src"},
)
