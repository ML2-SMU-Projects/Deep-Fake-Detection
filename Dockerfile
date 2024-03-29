# Image pulled and modified from this repo
# https://github.com/anibali/docker-pytorch/blob/master/dockerfiles/1.5.0-cuda9.2-ubuntu18.04/Dockerfile

FROM nvidia/cuda:9.2-base-ubuntu18.04

LABEL maintainer="Kay-Ayala@protonmail.com"

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \ 
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
 && rm -rf /var/lib/apt/lists/*

# Create a working directory
RUN mkdir /app
WORKDIR /app

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
 && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user

# Install Miniconda and Python 3.8
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV PATH=/home/user/miniconda/bin:$PATH
RUN curl -sLo ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-py38_4.8.2-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh \
 && conda install -y python==3.8.1 \
 && conda clean -ya

# CUDA 9.2-specific steps
RUN conda install -y -c pytorch \
    cudatoolkit=9.2 \
    "pytorch=1.5.0=py3.8_cuda9.2.148_cudnn7.6.3_0" \
    "torchvision=0.6.0=py38_cu92" \
 && conda clean -ya

RUN mkdir /home/user/scripts
ADD scripts /home/user/scripts

RUN git clone https://github.com/ML2-SMU-Projects/Deep-Fake-Detection.git

# Set the default command to bash
CMD ["bash"]
