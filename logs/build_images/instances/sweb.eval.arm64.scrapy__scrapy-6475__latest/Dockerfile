FROM --platform=linux/arm64/v8 sweb.env.arm64.9361a9629f2dd7af5c81e5:latest

COPY ./setup_repo.sh /root/
RUN /bin/bash /root/setup_repo.sh

WORKDIR /testbed/
