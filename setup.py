import setuptools
with open("README.md", "r") as fh:
    setuptools.setup(
        name='open-nre',  
        version='0.1.1',
        author="Tianyu Gao",
        author_email="gaotianyu1350@126.com",
        description="An open source toolkit for relation extraction",
        url="https://github.com/thunlp/opennre",
        packages=setuptools.find_packages(),
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: POSIX :: Linux",
        ],
        install_requires=['nltk', 'transformers', 'scikit-learn'],
        setup_requires=['wheel']
     )
