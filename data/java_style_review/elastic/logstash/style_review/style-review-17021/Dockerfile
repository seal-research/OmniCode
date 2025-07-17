FROM openjdk:17-slim


# Install necessary tools
RUN apt-get update && apt-get install -y wget unzip git jq bc

# Install Checkstyle
RUN wget -q https://github.com/checkstyle/checkstyle/releases/download/checkstyle-10.12.1/checkstyle-10.12.1-all.jar -O /usr/local/lib/checkstyle.jar

# Set up working directory
WORKDIR /workspace

# Copy configuration files
COPY checkstyle.xml /workspace/
COPY run_style_review.sh /workspace/
RUN chmod +x /workspace/run_style_review.sh


