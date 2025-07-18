FROM openjdk:17-slim


RUN apt-get update && apt-get install -y wget unzip git jq curl bc libxml2-utils

# Install PMD
RUN curl -L https://github.com/pmd/pmd/releases/download/pmd_releases%2F7.0.0/pmd-dist-7.0.0-bin.zip -o /tmp/pmd.zip && \
    unzip /tmp/pmd.zip -d /opt && \
    mv /opt/pmd-bin-7.0.0 /opt/pmd && \
    ln -s /opt/pmd/bin/pmd /usr/local/bin/pmd && \
    chmod +x /usr/local/bin/pmd

WORKDIR /workspace

COPY pmd-ruleset.xml /workspace/
COPY run_style_review.sh /workspace/
RUN chmod +x /workspace/run_style_review.sh


