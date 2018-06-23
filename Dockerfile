FROM ubuntu:16.04

RUN apt update && apt install -y --allow-unauthenticated --no-install-recommends \
        build-essential apt-utils cmake git curl vim ca-certificates \
        libjpeg-dev libpng-dev python3.5 python3-pip python3-setuptools \
        libgtk3.0 libsm6 python3-venv cmake ffmpeg pkg-config \
        qtbase5-dev libqt5opengl5-dev libassimp-dev libpython3.5-dev \
        libboost-python-dev libtinyxml-dev
RUN rm -rf /var/lib/apt/lists/*
WORKDIR /workspace/
RUN python3 -m venv py36
RUN /bin/bash
RUN source /workspace/py36/bin/activate
RUN pip install --upgrade pip
#COPY requirements.txt requirements.txt
#RUN pip install -r requirements.txt
#RUN pip install pybullet
