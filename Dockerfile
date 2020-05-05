FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

RUN apt-get update &&\
    apt-get -y install build-essential yasm nasm \
    cmake unzip git wget tmux nano \
    sysstat libtcmalloc-minimal4 pkgconf autoconf libtool \
    python3 python3-pip python3-dev python3-setuptools \
    libsm6 libxext6 libxrender-dev &&\
    ln -s /usr/bin/python3 /usr/bin/python &&\
    ln -s /usr/bin/pip3 /usr/bin/pip &&\
    apt-get clean &&\
    apt-get autoremove &&\
    rm -rf /var/lib/apt/lists/* &&\
    rm -rf /var/cache/apt/archives/*

RUN pip3 install --no-cache-dir numpy==1.18.4

# Install PyTorch
RUN pip3 install --no-cache-dir \
    torch==1.5.0 \
    torchvision==0.6.0

# Install python ML packages
RUN pip3 install --no-cache-dir \
    opencv-python==4.2.0.34 \
    scipy==1.4.1 \
    matplotlib==3.2.1 \
    pandas==1.0.3 \
    notebook==6.0.3 \
    scikit-learn==0.22.2.post1 \
    scikit-image==0.16.2 \
    pytorch-argus==0.0.9 \
    albumentations==0.4.5

RUN pip install timm==0.1.26

ENV PYTHONPATH $PYTHONPATH:/workdir
ENV TORCH_HOME=/workdir/data/.torch

WORKDIR /workdir
