from setuptools import find_packages, setup

with open("README.md", "r") as f:
    long_description = f.read()
    descr_lines = long_description.split("\n")
    descr_no_gifs = []  # gifs are not supported on PyPI web page
    for dl in descr_lines:
        if not ("<img src=" in dl and "gif" in dl):
            descr_no_gifs.append(dl)
    long_description = "\n".join(descr_no_gifs)

setup(
    # Information
    name="HASARD",
    description="Egocentric 3D Safe Reinforcement Learning Benchmark",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version="0.1.0",
    url="https://github.com/TTomilin/HASARD",
    author="Tristan Tomilin",
    author_email='tristan.tomilin@hotmail.com',
    license="MIT",
    keywords=["safe rl", "reinforcement learning", "vizdoom", "benchmark", "safety"],
    install_requires=[
        "numpy>=1.18.1,<2.0",
        "torch>=1.9,<3.0,!=1.13.0",
        "gymnasium>=0.27,<1.0",
        "pyglet",  # gym dependency
        "tensorboard>=1.15.0",
        "tensorboardx>=2.0",
        "psutil>=5.7.0",
        "threadpoolctl>=2.0.0",
        "colorlog",
        # "faster-fifo>=1.4.2,<2.0",  <-- installed by signal-slot-mp
        "signal-slot-mp>=1.0.3,<2.0",
        "filelock",
        "opencv-python",
        "wandb>=0.12.9",
        "pandas",
        "vizdoom",
        "pillow",
        "matplotlib",
        "torchviz",
    ],
    package_dir={"": "./"},
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.8",
)
