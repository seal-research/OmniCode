FROM --platform=linux/arm64/v8 sweb.env.arm64.2d525b989e7eb5542ed7f3:latest

COPY ./setup_repo.sh /root/
RUN /bin/bash /root/setup_repo.sh

WORKDIR /testbed/
