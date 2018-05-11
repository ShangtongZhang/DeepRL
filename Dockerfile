FROM ubuntu:16.04

RUN apt update
RUN apt install -y --allow-unauthenticated --no-install-recommends \
        build-essential apt-utils cmake git curl vim ca-certificates \
        libjpeg-dev libpng-dev python3.5 python3-pip python3-setuptools \
        libgtk3.0 libsm6 python3-dev
RUN rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/
COPY deep_rl deep_rl
COPY setup.py setup.py
COPY examples.py examples.py
RUN pip3 install --upgrade pip
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN ln -s /usr/bin/pip3 /usr/bin/pip
RUN pip install -e .
RUN pip install pybullet
