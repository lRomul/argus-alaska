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
    torch==1.5.1 \
    torchvision==0.6.0

RUN git clone https://github.com/NVIDIA/apex &&\
    cd apex &&\
    git checkout 3bae8c83494184673f01f3867fa051518e930895 &&\
    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" . &&\
    cd .. && rm -rf apex

# Install python ML packages
RUN pip3 install --no-cache-dir \
    opencv-python==4.2.0.34 \
    scipy==1.4.1 \
    matplotlib==3.2.1 \
    pandas==1.0.3 \
    notebook==6.0.3 \
    scikit-learn==0.22.2.post1 \
    scikit-image==0.16.2 \
    albumentations==0.4.5 \
    adamp==0.2.0

RUN pip install --no-cache-dir \
    timm==0.1.26 \
    Cython==0.29.17

RUN pip install -U git+https://github.com/lRomul/argus.git@custom_build

RUN git clone -b master https://github.com/dwgoon/jpegio &&\
    cd jpegio &&\
    git checkout fe577469cc332c14f8647167ca8ca2b573f5071b &&\
    python setup.py install &&\
    cd .. && rm -rf jpegio

# RUN pip install git+https://github.com/mapillary/inplace_abn.git@v1.0.12

ENV PYTHONPATH $PYTHONPATH:/workdir
ENV TORCH_HOME=/workdir/data/.torch

WORKDIR /workdir
