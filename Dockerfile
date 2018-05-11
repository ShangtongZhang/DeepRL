FROM ubuntu:16.04

RUN apt update
RUN apt install -y --allow-unauthenticated --no-install-recommends \
        build-essential apt-utils cmake git curl vim ca-certificates \
        libjpeg-dev libpng-dev python3.5 python3-pip python3-setuptools \
        libgtk3.0 libsm6 python3-dev openssh-server
RUN rm -rf /var/lib/apt/lists/*

WORKDIR /tmp
RUN pip3 install --upgrade pip
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN ln -s /usr/bin/pip3 /usr/bin/pip

RUN mkdir /var/run/sshd
RUN echo 'root:deeprl' | chpasswd
RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# SSH login fix. Otherwise user is kicked off after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile

EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN pip install pybullet
WORKDIR /workspace/
