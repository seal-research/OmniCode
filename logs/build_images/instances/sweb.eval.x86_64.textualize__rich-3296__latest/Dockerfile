FROM --platform=linux/x86_64 sweb.env.x86_64.4ebcb0e7d8c9a5c9824428:latest

COPY ./setup_repo.sh /root/
# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
# Add poetry to the PATH
ENV PATH="$PATH:/root/.local/bin"
RUN /bin/bash /root/setup_repo.sh
WORKDIR /testbed/
