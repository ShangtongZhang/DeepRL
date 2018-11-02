FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04

# CUDA includes
ENV CUDA_PATH /usr/local/cuda
ENV CUDA_INCLUDE_PATH /usr/local/cuda/include
ENV CUDA_LIBRARY_PATH /usr/local/cuda/lib64

RUN echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list

ENV CUDNN_VERSION 6.0.20

RUN apt update && apt install -y --allow-unauthenticated --no-install-recommends \
    build-essential apt-utils cmake git curl vim ca-certificates \
    libjpeg-dev libpng-dev python3.5 python3-pip python3-setuptools \
    libgtk3.0 libsm6 python3-venv cmake ffmpeg pkg-config \
    qtbase5-dev libqt5opengl5-dev libassimp-dev libpython3.5-dev \
    libboost-python-dev libtinyxml-dev bash python3-tk libcudnn6=$CUDNN_VERSION-1+cuda8.0 \
    libcudnn6-dev=$CUDNN_VERSION-1+cuda8.0 wget unzip libosmesa6-dev software-properties-common \
    libopenmpi-dev
WORKDIR /opt
RUN pip3 install pip --upgrade

RUN add-apt-repository ppa:jamesh/snap-support && apt-get update && apt install -y patchelf
RUN mkdir -p /root/.mujoco \
    && wget https://www.roboti.us/download/mjpro150_linux.zip -O mujoco.zip \
    && unzip mujoco.zip -d /root/.mujoco \
    && rm mujoco.zip
RUN wget https://www.roboti.us/download/mujoco200_linux.zip -O mujoco.zip \
    && unzip mujoco.zip -d /root/.mujoco \
    && rm mujoco.zip

RUN export PATH=$PATH:$HOME/.local/bin
# Make sure you have the license
COPY ./mjkey.txt /root/.mujoco/mjkey.txt
ENV LD_LIBRARY_PATH /root/.mujoco/mjpro150/bin:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH /root/.mujoco/mjpro200_linux/bin:${LD_LIBRARY_PATH}

RUN pip3 install gym[mujoco] --upgrade
RUN pip3 install mujoco-py
RUN echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mjpro150/bin" >> ~/.bashrc
RUN echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mjpro200_linux/bin" >> ~/.bashrc

RUN git clone https://github.com/openai/roboschool.git
ENV ROBOSCHOOL_PATH=/opt/roboschool
RUN git clone https://github.com/olegklimov/bullet3 -b roboschool_self_collision
WORKDIR /opt/bullet3/build
RUN cmake -DBUILD_SHARED_LIBS=ON -DUSE_DOUBLE_PRECISION=1 -DCMAKE_INSTALL_PREFIX:PATH=$ROBOSCHOOL_PATH/roboschool/cpp-household/bullet_local_install -DBUILD_CPU_DEMOS=OFF -DBUILD_BULLET2_DEMOS=OFF -DBUILD_EXTRAS=OFF  -DBUILD_UNIT_TESTS=OFF -DBUILD_CLSOCKET=OFF -DBUILD_ENET=OFF -DBUILD_OPENGL3_DEMOS=OFF ..
RUN make -j4
RUN make install
RUN pip3 install -e $ROBOSCHOOL_PATH

# install other requirements
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
RUN pip3 install git+git://github.com/openai/baselines.git@8e56dd#egg=baselines

WORKDIR /workspace/DeepRL
RUN rm -rf /var/lib/apt/lists/*
