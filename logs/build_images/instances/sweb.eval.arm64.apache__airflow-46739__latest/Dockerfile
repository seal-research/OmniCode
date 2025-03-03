FROM --platform=linux/arm64/v8 sweb.env.arm64.6cfd86235b09fe9cee33c2:latest

COPY ./setup_repo.sh /root/
RUN /bin/bash /root/setup_repo.sh

WORKDIR /testbed/
