FROM ubuntu:16.04

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y \
    build-essential \
    ca-certificates \
    gcc \
    git \
    sudo \
    wget \
    libpq-dev \
    make \
    ssh \
    && apt-get autoremove \
    && apt-get clean

# RUN apt-get install tshark -y

RUN apt-get install software-properties-common -y
RUN add-apt-repository ppa:webupd8team/java
RUN apt-get update
RUN echo oracle-java8-installer shared/accepted-oracle-license-v1-1 select true | /usr/bin/debconf-set-selections
RUN apt-get install oracle-java8-installer libpcap-dev -y

RUN apt-get install software-properties-common -y
RUN add-apt-repository ppa:linuxuprising/java
RUN echo debconf shared/accepted-oracle-license-v1-2 select true | debconf-set-selections
RUN echo debconf shared/accepted-oracle-license-v1-2 seen true | debconf-set-selections
RUN apt-get update
RUN apt-get install -y libpcap-dev
RUN apt-get install oracle-java14-set-default -y

# Establish working directory
WORKDIR /CICFlowMeter

