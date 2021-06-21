import setuptools


def read(filename: str) -> str:
    with open(filename, "r") as f:
        return f.read()


setuptools.setup(
    name="opennre",
    version="0.1",
    author="Tianyu Gao",
    author_email="gaotianyu1350@126.com",
    description="An open source toolkit for relation extraction",
    url="https://github.com/thunlp/opennre",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Linux",
    ],
    setup_requires=["wheel"],
    install_requires=read("requirements.txt").strip().splitlines(),
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    license_files=("LICENSE",),
)
