import setuptools

with open("README.md") as file:
    long_description = file.read()

setuptools.setup(
    name="torchfit",
    version="0.0.1",
    packages=setuptools.find_packages(),
    install_requires=[
        'torch',
    ],
    author="Holim Lim",
    author_email="ihl7029@europa.snu.ac.kr",
    description="torch boilerplate",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Holim0711/torchfit",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
