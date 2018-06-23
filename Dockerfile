FROM ubuntu:16.04

RUN apt update && apt install -y --allow-unauthenticated --no-install-recommends \
    build-essential apt-utils cmake git curl vim ca-certificates \
    libjpeg-dev libpng-dev python3.5 python3-pip python3-setuptools \
    libgtk3.0 libsm6 python3-venv cmake ffmpeg pkg-config \
    qtbase5-dev libqt5opengl5-dev libassimp-dev libpython3.5-dev \
    libboost-python-dev libtinyxml-dev bash
WORKDIR /opt
SHELL ["/bin/bash", "-c"]
RUN pip3 install pip --upgrade

# install Roboschool
RUN git clone https://github.com/openai/roboschool.git
ENV ROBOSCHOOL_PATH=/opt/roboschool
RUN git clone https://github.com/olegklimov/bullet3 -b roboschool_self_collision
WORKDIR /opt/bullet3/build
RUN cmake -DBUILD_SHARED_LIBS=ON -DUSE_DOUBLE_PRECISION=1 -DCMAKE_INSTALL_PREFIX:PATH=$ROBOSCHOOL_PATH/roboschool/cpp-household/bullet_local_install -DBUILD_CPU_DEMOS=OFF -DBUILD_BULLET2_DEMOS=OFF -DBUILD_EXTRAS=OFF  -DBUILD_UNIT_TESTS=OFF -DBUILD_CLSOCKET=OFF -DBUILD_ENET=OFF -DBUILD_OPENGL3_DEMOS=OFF ..
RUN make -j4
RUN make install
WORKDIR /workspace
RUN pip3 install -e $ROBOSCHOOL_PATH

# install PyBullet
RUN pip3 install pybullet

# install other requirements
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

RUN rm -rf /var/lib/apt/lists/*
RUN rm -f /usr/bin/python && ln -s /usr/bin/python3 /usr/bin/python
RUN rm -f /usr/bin/pip && ln -s /usr/bin/pip3 /usr/bin/pip
