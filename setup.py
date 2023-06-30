#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author     : Lin Cong
# E-mail     : cong@informatik.uni-hamburg.de
# Description: Reinforcement Learning Method for Object Pushing
# Date       : 11/03/2021
# File Name  : setup.py
from setuptools import setup, find_packages

__version__ = "0.0.1"
setup(
    name="visual-pushing",
    version=__version__,
    keywords=["manipulation", "deep-reinforcement-learning"],
    description="pushing object based on visual information",
    license="MIT License",
    url="https://tams.informatik.uni-hamburg.de/",
    author="Lin Cong",
    author_email="cong@informatik.uni-hamburg.de",
    packages=find_packages(),
    include_package_data=True,
    platforms="any",
    install_requires=[]
)
