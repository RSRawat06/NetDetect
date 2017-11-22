From tensorflow/tensorflow:latest-py3

# Install necessary packages
RUN apt update
RUN apt install gcc make libc-dev g++ bzip2 libssl-dev build-essential wget -y

# Hop into /NetDetect
WORKDIR /NetDetect

# Install Python requirements
COPY ./NetDetect/requirements.txt ./requirements.txt
RUN export C_INCLUDE_PATH=/usr/include
RUN pip3 install --upgrade pip
RUN pip3 install -r ./requirements.txt

# Hop into /
WORKDIR /

