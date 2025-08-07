FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

WORKDIR /home/

# Install base packages
RUN apt-get update && \
    apt-get install -y git openjdk-17-jdk maven curl python3 python3-pip python3-venv && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set correct PATH for root pipx installs
ENV PATH="/root/.local/bin:$PATH"

# Install pipx globally for root user and ensure it's in path
RUN pip3 install --user pipx && \
    /root/.local/bin/pipx ensurepath && \
    /root/.local/bin/pipx install swe-rex

# Clone repo (optional for test)
RUN git clone https://github.com/elastic/logstash.git /home/logstash

