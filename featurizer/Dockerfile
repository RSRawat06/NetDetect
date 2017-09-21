FROM ubuntu:16.04

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y \
    build-essential \
    ca-certificates \
    gcc \
    git \
    libpq-dev \
    make \
    python-pip \
    python2.7 \
    python2.7-dev \
    ssh \
    && apt-get autoremove \
    && apt-get clean

# RUN apt-get install tshark -y

RUN pip install -U "setuptools==3.4.1"
RUN pip install -U "pip==1.5.4"

# Establish working directory
WORKDIR /service

# Copying pip requirement files
COPY ./featurizer/requirements.txt ./requirements.txt

# Install python dependencies
RUN export C_INCLUDE_PATH=/usr/include
RUN pip install --upgrade pip
RUN pip install -r ./requirements.txt
