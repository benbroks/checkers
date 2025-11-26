from setuptools import setup, find_packages

setup(
    name="python-checkers",
    version="1.0.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "pygame>=2.0.0",
    ],
    entry_points={
        "console_scripts": [
            "checkers-gui=checkers.gui.game_control:main",
            "checkers-cli=checkers.cli.interface:main",
        ],
    },
    python_requires=">=3.7",
    author="Python Checkers Team",
    description="A checkers game implementation with AI support",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
