import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()
    setuptools.setup(
        name='opennre',  
        version='0.1',
        author="Tianyu Gao",
        author_email="gaotianyu1350@126.com",
        description="An open source toolkit for relation extraction",
        long_description=long_description,
        url="https://github.com/thunlp/opennre",
        packages=setuptools.find_packages(),
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ],
     )
