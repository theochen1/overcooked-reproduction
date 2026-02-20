FROM --platform=linux/amd64 ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.7 \
    python3.7-dev \
    python3-pip \
    git \
    wget \
    build-essential \
    libopenmpi-dev \
    openmpi-bin \
    libsdl1.2-dev \
    libsdl-image1.2-dev \
    libsdl-mixer1.2-dev \
    libsdl-ttf2.0-dev \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.7 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.7 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Upgrade pip
RUN pip install --upgrade pip==19.3.1 setuptools==42.0.0 wheel

# Install conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.3-Linux-x86_64.sh -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

ENV PATH=/opt/conda/bin:$PATH

# Create conda environment
RUN conda create -n harl python=3.7 -y

SHELL ["/bin/bash", "-c"]

WORKDIR /workspace

# Clone with submodules
RUN git clone --recursive https://github.com/HumanCompatibleAI/human_aware_rl.git

WORKDIR /workspace/human_aware_rl

# CRITICAL: Install TensorFlow 1.13.1 BEFORE install.sh to prevent TF 2.x
RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda activate harl && \
    pip install tensorflow==1.13.1

# Pre-install compatible versions to prevent install.sh from pulling wrong ones
RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda activate harl && \
    pip install \
        numpy==1.16.4 \
        pymongo==3.9.0 \
        pandas==1.0.5 \
        matplotlib==3.0.3 \
        seaborn==0.9.0 \
        pygame==1.9.6 \
        GitPython==3.0.5 \
        memory-profiler==0.57.0 \
        sacred==0.7.4

# Make install.sh executable
RUN chmod +x install.sh

# Now run install.sh (it will skip already-installed packages)
RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda activate harl && \
    ./install.sh

# Force reinstall TensorFlow 1.13.1 in case install.sh changed it
RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda activate harl && \
    pip install --force-reinstall --no-deps tensorflow==1.13.1

# Install mpi4py via conda (as per README)
RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda activate harl && \
    conda install mpi4py==3.0.3 -y

# Set up environment variables
ENV PYTHONPATH=/workspace/human_aware_rl:/workspace/human_aware_rl/overcooked_ai

# Activate conda environment by default
RUN echo "source /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate harl" >> ~/.bashrc

WORKDIR /workspace/human_aware_rl/human_aware_rl

CMD ["/bin/bash"]
