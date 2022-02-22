#! -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name='tda',
    version='1.0',
    author='xiangbei',
    packages=find_packages(),
    install_requires=['jieba', 'tqdm'],
    package_data={
        '': ['*.txt', '*.pkl']
    }
)
