FROM --platform=linux/arm64/v8 sweb.env.arm64.881df5e55234403a125b3e:latest

COPY ./setup_repo.sh /root/
RUN /bin/bash /root/setup_repo.sh

WORKDIR /testbed/
