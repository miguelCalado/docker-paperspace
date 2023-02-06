#!/bin/bash

# Paperspace Dockerfile for Gradient base image
# Paperspace image is located in Dockerhub registry: paperspace/gradient_base


# ==================================================================
# Initial setup
# ------------------------------------------------------------------

    # Ubuntu 20.04 as base image
    FROM ubuntu:20.04
    RUN yes| unminimize

    # Set ENV variables
    ENV LANG C.UTF-8
    ENV SHELL=/bin/bash
    ENV DEBIAN_FRONTEND=noninteractive

    ENV APT_INSTALL="apt-get install -y --no-install-recommends"
    ENV PIP_INSTALL="python3 -m pip --no-cache-dir install"
    ENV GIT_CLONE="git clone --depth 10"


# ==================================================================
# Tools
# ------------------------------------------------------------------

    RUN apt-get update && \
        $APT_INSTALL \
        apt-utils \
        gcc \
        make \
        pkg-config \
        apt-transport-https \
        build-essential \
        ca-certificates \
        wget \
        rsync \
        git \
        vim \
        mlocate \
        libssl-dev \
        curl \
        openssh-client \
        unzip \
        unrar \
        zip \
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
        libboost-all-dev \
        cifs-utils \
        software-properties-common \
        cron \
        tmux


# ==================================================================
# Python
# ------------------------------------------------------------------

    #Based on https://launchpad.net/~deadsnakes/+archive/ubuntu/ppa

    # Adding repository for python3.9
    RUN add-apt-repository ppa:deadsnakes/ppa -y && \

    # Installing python3.9
        $APT_INSTALL \
        python3.9 \
        python3.9-dev \
        python3.9-venv \
        python3-distutils-extra

    # Add symlink so python and python3 commands use same python3.9 executable
    RUN ln -s /usr/bin/python3.9 /usr/local/bin/python3 && \
        ln -s /usr/bin/python3.9 /usr/local/bin/python && \

    # Remove `site-packages` and soft link it to `dist-packages` to allow `pip install -e .`
    # Source: https://stackoverflow.com/a/72962729/8527630
    rm -rf /usr/lib/python3.9/site-packages/ && \
    ln -s /usr/local/lib/python3.9/dist-packages /usr/lib/python3.9/site-packages

    # Installing pip
    RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.9
    ENV PATH=$PATH:/root/.local/bin


# ==================================================================
# Installing CUDA packages (CUDA Toolkit 11.6.2 & CUDNN 8.4.1)
# ------------------------------------------------------------------

    # Based on https://developer.nvidia.com/cuda-toolkit-archive
    # Based on https://developer.nvidia.com/rdp/cudnn-archive
    # Based on https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#package-manager-ubuntu-install

    # Installing CUDA Toolkit
    RUN wget https://developer.download.nvidia.com/compute/cuda/11.6.2/local_installers/cuda_11.6.2_510.47.03_linux.run && \
        bash cuda_11.6.2_510.47.03_linux.run --silent --toolkit && \
        rm cuda_11.6.2_510.47.03_linux.run
    ENV PATH=$PATH:/usr/local/cuda-11.6/bin
    ENV LD_LIBRARY_PATH=/usr/local/cuda-11.6/lib64

    # Installing CUDNN
    RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin && \
        mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600 && \
        apt-get install dirmngr -y && \
        apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub && \
        add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /" && \
        apt-get update && \
        apt-get install libcudnn8=8.4.1.*-1+cuda11.6 -y && \
        apt-get install libcudnn8-dev=8.4.1.*-1+cuda11.6 -y && \
        rm /etc/apt/preferences.d/cuda-repository-pin-600

# ==================================================================
# CMake
# ------------------------------------------------------------------

    RUN $GIT_CLONE https://github.com/Kitware/CMake ~/cmake && \
        cd ~/cmake && \
        ./bootstrap && \
        make -j"$(nproc)" install


# ==================================================================
# Installing JRE and JDK
# ------------------------------------------------------------------

    RUN $APT_INSTALL \
        default-jre \
        default-jdk

# ==================================================================
# PyTorch
# ------------------------------------------------------------------

    # Based on https://pytorch.org/get-started/locally/

    RUN $PIP_INSTALL torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116 && \
        

# ==================================================================
# JAX
# ------------------------------------------------------------------

    # Based on https://github.com/google/jax#pip-installation-gpu-cuda

    $PIP_INSTALL "jax[cuda11_cudnn82]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html && \
    $PIP_INSTALL flax==0.6.3 && \


# ==================================================================
# TensorFlow
# ------------------------------------------------------------------

    # Based on https://www.tensorflow.org/install/pip

    $PIP_INSTALL tensorflow==2.11.0 && \


# ==================================================================
# Hugging Face
# ------------------------------------------------------------------
    
    # Based on https://huggingface.co/docs/transformers/installation
    # Based on https://huggingface.co/docs/datasets/installation

    $PIP_INSTALL transformers==4.21.3 datasets==2.4.0 && \

# ==================================================================
# Additional Python Packages
# ------------------------------------------------------------------

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
        seaborn==0.12.0 \
        SQLAlchemy==1.4.39 \
        spacy==3.4.0 \
        nltk==3.7 \
        jsonify==0.5 \
        boto3==1.24.27 \
        transformers==4.22.1 \
        sentence-transformers==2.2.2 \
        datasets==2.3.2 \
        opencv-python==4.6.0.66 \
        msal \
        elementpath \
        lxml==4.9.1 \
        wandb \
        jupyter_resource_usage \
        types-requests \
        pytest \
        isort \
        black \
        flake8 \
        mypy \
        pyopenssl \
        cdifflib \
        nbqa \
        colour \
        pycocotools \
        tensorflow_datasets

# ==================================================================
# JupyterLab & Notebook
# ------------------------------------------------------------------

    # Based on https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html#pip

    RUN $PIP_INSTALL \
            jupyterlab==3.4.6 \
            jupyter==1.0.0 \
            notebook==6.4.12


# ==================================================================
# Node.js and Jupyter Notebook Extensions
# ------------------------------------------------------------------

    # Install dependencies for running Jupyter Notebook/Lab with extensions
    RUN curl -sL https://deb.nodesource.com/setup_16.x | bash  && \
        $APT_INSTALL nodejs  && \
        $PIP_INSTALL \
            jupyter_contrib_nbextensions==0.5.1 \
            jupyter_nbextensions_configurator==0.4.1

    # Enable nbextensions and menubar
    RUN jupyter nbextensions_configurator enable --user && \
        jupyter contrib nbextension install --user

    # Enable Jupyter Notebook extensions
    RUN jupyter nbextension enable spellchecker/main && \
        jupyter nbextension enable snippets_menu/main && \
        jupyter nbextension enable snippets/main && \
        jupyter nbextension enable freeze/main && \
        jupyter nbextension enable livemdpreview/livemdpreview && \
        jupyter nbextension enable highlight_selected_word/main && \
        jupyter nbextension enable execute_time/ExecuteTime && \
        jupyter nbextension enable toc2/main && \
        jupyter nbextension enable jupyter_resource_usage/main

# ==================================================================
# Add Jupyter Notebook configurations
# ------------------------------------------------------------------

    # Enable CPU usage
    RUN jupyter notebook --generate-config && \
        echo "c.ResourceUseDisplay.track_cpu_percent = True" >> ~/.jupyter/jupyter_notebook_config.py && \
        echo "c.ResourceUseDisplay.enable_prometheus_metrics = False" >> ~/.jupyter/jupyter_notebook_config.py

    # Get predefined extensions and macros
    COPY notebook.json ./
    RUN rm ~/.jupyter/nbconfig/notebook.json && mv ./notebook.json ~/.jupyter/nbconfig/


# ==================================================================
# Startup
# ------------------------------------------------------------------

    EXPOSE 8888 6006

    CMD jupyter notebook --allow-root --ip=0.0.0.0 --no-browser --ServerApp.trust_xheaders=True --ServerApp.disable_check_xsrf=False --ServerApp.allow_remote_access=True --ServerApp.allow_origin='*' --ServerApp.allow_credentials=True