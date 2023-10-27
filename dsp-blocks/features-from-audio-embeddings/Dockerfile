# syntax = docker/dockerfile:experimental
FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# python3 and all dependencies for scipy
RUN apt update && apt install -y python3 python3-pip libatlas-base-dev gfortran-9 libfreetype6-dev && \
    ln -s $(which gfortran-9) /usr/bin/gfortran

# Update pip
RUN pip3 install -U pip==22.0.3

# Cython and scikit-learn - it needs to be done in this order for some reason
RUN pip3 --no-cache-dir install Cython==0.29.24

# Install TensorFlow (separate script as this requires a different command on M1 Macs)
COPY build-scripts/install_tensorflow.sh install_tensorflow.sh
RUN /bin/bash install_tensorflow.sh && \
    rm install_tensorflow.sh

# Rest of the dependencies
COPY requirements-blocks.txt ./
RUN pip3 --no-cache-dir install -r requirements-blocks.txt

COPY third_party /third_party
COPY . ./

EXPOSE 4446

ENTRYPOINT ["python3", "-u", "dsp-server.py"]
