FROM --platform=linux/x86_64 sweb.env.x86_64.d24dbffb3ac01f5e71750c:latest

COPY ./setup_repo.sh /root/
RUN /bin/bash /root/setup_repo.sh

WORKDIR /testbed/
