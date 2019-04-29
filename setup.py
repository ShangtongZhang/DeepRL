from setuptools import setup, find_packages
import sys

print('Please install OpenAI Baselines (commit 8e56dd) and requirement.txt')
if not (sys.version.startswith('3.5') or sys.version.startswith('3.6')):
    raise Exception('Only Python 3.5 and 3.6 are supported')

setup(name='deep_rl',
      packages=[package for package in find_packages()
                if package.startswith('deep_rl')],
      install_requires=[],
      description="Modularized Implementation of Deep RL Algorithms",
      author="Shangtong Zhang",
      url='https://github.com/ShangtongZhang/DeepRL',
      author_email="zhangshangtong.cpp@gmail.com",
      version="1.1")