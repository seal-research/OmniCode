FROM ubuntu:22.04



ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

WORKDIR /home/
RUN apt-get update && apt-get install -y git openjdk-17-jdk maven
# install swe-rex for sweagent:
RUN apt-get update && apt-get install -y python3 python3-pip python3-venv
RUN pip3 install pipx &&     pipx ensurepath &&     pipx install swe-rex

RUN git clone https://github.com/google/guava.git /home/guava


