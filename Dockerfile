# Paperspace Dockerfile for Gradient base image
# Paperspace image is located in Dockerhub registry: miguelcalado/docker-paperspace
# or in https://hub.docker.com/repository/docker/miguelcalado/docker-paperspace

# ==================================================================
# module list
# ------------------------------------------------------------------
# python                 3.9.13   (apt)
# jupyter                latest   (pip)
# pytorch                latest   (pip)
# tensorflow             2.9.1    (pip)
# tensorflow_addons      latest   (pip)
# ftfy                   latest   (pip)
# jupyterlab             latest   (pip)
# keras                  latest   (pip) # Comes installed with Tensorflow
# opencv                 4.5.1    (git)
# numpy                  1.23.1   (pip)
# scipy                  1.8.1    (pip)
# pandas                 1.4.3    (pip)
# cloudpickle            2.1.0    (pip)
# scikit-image           0.19.3   (pip)
# scikit-learn           1.1.1    (pip)
# matplotlib             3.5.2    (pip)
# ipython                8.4.0    (pip)
# ipykernel              6.15.1   (pip)
# ipywidgets             7.7.1    (pip)
# gradient               2.0.5    (pip)
# Cython                 0.29.30  (pip)
# tqdm                   4.64.0   (pip)
# gdown                  4.5.1    (pip)
# xgboost                1.6.1    (pip) 
# pillow                 latest   (pip)
# seaborn                0.11.2   (pip)
# SQLAlchemy             1.4.39   (pip)
# spacy                  3.4.0    (pip)
# nltk                   3.7      (pip)
# jsonify                0.5      (pip)
# boto3                  1.24.27  (pip)
# transformers           4.20.1   (pip)
# sentence-transformers  2.2.2    (pip)
# opencv-python          4.6.0.66 (pip)
# JAX 		             latest   (pip) # Currently not installed
# JAXlib 	             latest   (pip) # Currently not installed
# msal                   latest   (pip)
# lxml                   4.9.1    (pip)
# wandb                  latest   (pip)
# jupyter_resource_usage latest   (pip)
# ==================================================================

# Ubuntu 20.04, CUDA Toolkit 11.2, CUDNN 8

FROM nvcr.io/nvidia/cuda:11.2.1-cudnn8-devel-ubuntu20.04
ENV LANG C.UTF-8

# Setting shell to bash
ENV SHELL=/bin/bash
SHELL ["/bin/bash", "-c"]

RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="pip --no-cache-dir install" && \
    GIT_CLONE="git clone --depth 10" && \

    rm -rf /var/lib/apt/lists/* \
           /etc/apt/sources.list.d/cuda.list \
           /etc/apt/sources.list.d/nvidia-ml.list && \

    apt-get update && \

# ==================================================================
# tools
# ------------------------------------------------------------------

    sed -e '\|/usr/share/man|s|^#*|#|g' -i /etc/dpkg/dpkg.cfg.d/excludes && \

    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        build-essential \
        apt-utils \
        ca-certificates \
        wget \
        rsync \
        git \
        vim \
        libssl-dev \
        curl \
        openssh-client \
        unzip \
        unrar \
        zip \
        awscli \
        csvkit \
        emacs \
        joe \
        jq \
        dialog \
        man-db \
        manpages \
        manpages-dev \
        manpages-posix \
        manpages-posix-dev \
        nano \
        iputils-ping \
        sudo \
        ffmpeg \
        libsm6 \
        libxext6 \
        && \

    rm -f /usr/bin/man && \
    dpkg-divert --quiet --remove --rename /usr/bin/man && \
    rm -f /usr/share/man/man1/sh.1.gz && \
    dpkg-divert --quiet --remove --rename /usr/share/man/man1/sh.1.gz && \

    $GIT_CLONE https://github.com/Kitware/CMake ~/cmake && \
    cd ~/cmake && \
    ./bootstrap && \
    make -j"$(nproc)" install && \


# ==================================================================
# Python
# ------------------------------------------------------------------

    # Installing python3.9
    DEBIAN_FRONTEND=noninteractive \
    $APT_INSTALL software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa -y && \

    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
    python3.9 \
    python3.9-dev \
    python3-distutils-extra \
    && \

    # Installing pip
    wget -O ~/get-pip.py \
    https://bootstrap.pypa.io/get-pip.py && \
    python3.9 ~/get-pip.py && \

    # Add symlink so python and python3 commands use same
    # python3.9 executable
    ln -s /usr/bin/python3.9 /usr/local/bin/python3 && \
    ln -s /usr/bin/python3.9 /usr/local/bin/python && \

    # Intalling Python packages
    $PIP_INSTALL \
        numpy==1.23.1 \
        scipy==1.8.1 \
        pandas==1.4.3 \
        cloudpickle==2.1.0 \
        scikit-image==0.19.3 \
        scikit-learn==1.1.1 \
        matplotlib==3.5.2 \
        ipython==8.4.0 \
        ipykernel==6.15.1 \
        ipywidgets==7.7.1 \
        gradient==2.0.5 \
        Cython==0.29.30 \
        tqdm==4.64.0 \
        gdown==4.5.1 \
        xgboost==1.6.1 \ 
        pillow \
        seaborn==0.11.2 \
        SQLAlchemy==1.4.39 \
        spacy==3.4.0 \
        nltk==3.7 \
        jsonify==0.5 \
        boto3==1.24.27 \
        transformers==4.20.1 \
        sentence-transformers==2.2.2 \
        datasets==2.3.2 \
        opencv-python==4.6.0.66 \
        msal \
        elementpath \
        lxml==4.9.1 \
        wandb \
        jupyter_resource_usage \
        && \


# ==================================================================
# boost
# ------------------------------------------------------------------

    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        libboost-all-dev \
        && \


# ==================================================================
# jupyter
# ------------------------------------------------------------------

    $PIP_INSTALL \
        jupyter \
        && \


# ==================================================================
# PyTorch
# ------------------------------------------------------------------

    $PIP_INSTALL torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116 && \
        

# ==================================================================
# TensorFlow
# ------------------------------------------------------------------

    # Based on https://www.tensorflow.org/install and 
    # https://www.tensorflow.org/install/pip, so is now not -gpu

    $PIP_INSTALL \
        tensorflow==2.9.1 \
        tensorflow_addons \
        ftfy \
        && \

# ==================================================================
# JAX
# ------------------------------------------------------------------


    # $PIP_INSTALL \
    #     "jax[cuda111]" -f https://storage.googleapis.com/jax-releases/jax_releases.html \
    #     https://storage.googleapis.com/jax-releases/cuda11/jaxlib-0.3.8+cuda11.cudnn82-cp39-none-manylinux2014_x86_64.whl \
    #     && \

# ==================================================================
# JupyterLab
# ------------------------------------------------------------------

    $PIP_INSTALL \
        jupyterlab \
        && \


# ==================================================================
# Node.js and Jupyter Notebook Extensions
# ------------------------------------------------------------------

    curl -sL https://deb.nodesource.com/setup_16.x | bash && \
    $APT_INSTALL nodejs && \
    $PIP_INSTALL jupyter_contrib_nbextensions jupyterlab-git && \
    jupyter contrib nbextension install --sys-prefix && \
    jupyter nbextension enable highlight_selected_word/main && \
    jupyter nbextension enable execute_time/ExecuteTime && \
    jupyter nbextension enable toc2/main && \
    jupyter nbextension enable jupyter_resource_usage/main && \

# ==================================================================
# Conda
# ------------------------------------------------------------------

    # wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    # /bin/bash ~/miniconda.sh -b -p /opt/conda && \


# ==================================================================
# Config & Cleanup
# ------------------------------------------------------------------

    ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/*

# ==================================================================
# Add Jupyter Notebook configurations
# ------------------------------------------------------------------

COPY notebook.json ./
RUN rm ~/.jupyter/nbconfig/notebook.json && mv ./notebook.json ~/.jupyter/nbconfig/

EXPOSE 8888 6006