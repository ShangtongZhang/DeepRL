from setuptools import setup, find_packages

setup(name='deeprl',
      packages=[package for package in find_packages()
                if package.startswith('deeprl')],
      install_requires=[
          'gym[mujoco,atari,classic_control]',
      ],
      description="Highly modularized implementation of popular deep RL algorithms",
      author="ShangtongZhang",
      url='https://github.com/ShangtongZhang/DeepRL',
      author_email="zhangshangtong.cpp@gmail.com",
      version="0.0.1")
