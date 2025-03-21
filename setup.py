from setuptools import setup, find_packages

setup(
    name="lean-mm",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A brief description of the lean-mm project.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yuanqing-wang/lean-mm",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        # Add your project's dependencies here
        # Example: 'numpy>=1.21.0',
    ],
    entry_points={
        "console_scripts": [
            # Example: 'lean-mm=lean_mm.cli:main',
        ],
    },
)