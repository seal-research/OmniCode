FROM --platform=linux/arm64/v8 sweb.env.arm64.71be0585f155067b3e5442:latest

COPY ./setup_repo.sh /root/
RUN /bin/bash /root/setup_repo.sh

WORKDIR /testbed/
