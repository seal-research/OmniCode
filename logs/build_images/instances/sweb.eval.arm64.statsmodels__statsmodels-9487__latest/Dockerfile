FROM --platform=linux/arm64/v8 sweb.env.arm64.be8913e8ad98c17f39a595:latest

COPY ./setup_repo.sh /root/
RUN /bin/bash /root/setup_repo.sh

WORKDIR /testbed/
