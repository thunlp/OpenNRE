import setuptools
with open("README.md", "r") as fh:
    setuptools.setup(
        name='opennre',  
        version='0.1',
        packages=setuptools.find_packages(),
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: Linux",
        ],
        setup_requires=['wheel']
     )
