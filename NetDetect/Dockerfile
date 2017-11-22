From tensorflow/tensorflow:latest-py3

# Install necessary packages
RUN apt update
RUN apt install gcc make libc-dev g++ bzip2 libssl-dev build-essential wget -y

# Hop into /NetDetect
WORKDIR /NetDetect

# Install Java 8 (Neo4j dependency)
RUN add-apt-repository ppa:webupd8team/java
RUN apt-get update
RUN echo oracle-java8-installer shared/accepted-oracle-license-v1-1 select true | /usr/bin/debconf-set-selections
RUN apt-get install oracle-java8-installer -y

# Install Python requirements
COPY ./NetDetect/requirements.txt ./requirements.txt
RUN export C_INCLUDE_PATH=/usr/include
RUN pip3 install --upgrade pip
RUN pip3 install -r ./requirements.txt

# Hop into /
WORKDIR /

