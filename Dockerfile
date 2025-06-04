# Base image with Ubuntu 22.04 and CUDA 12.6
FROM nvidia/cuda:12.6.2-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt update && apt install -y \
    wget curl git unzip \
    python3 python3-pip \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda for Conda environments
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda.sh && \
    bash /miniconda.sh -b -p /miniconda && \
    rm /miniconda.sh && \
    /miniconda/bin/conda init bash

# Set environment variables for Conda
ENV PATH="/miniconda/bin:$PATH"
SHELL ["/bin/bash", "--login", "-c"]

# Copy YAML environment files into the container
COPY cihp.yaml /tmp/cihp.yaml
COPY StableVITON.yaml /tmp/StableVITON.yaml
COPY test.yaml /tmp/test.yaml

# Create Conda environments from YAML files
RUN conda env create -f /tmp/cihp.yaml
RUN conda env create -f /tmp/StableVITON.yaml
RUN conda env create -f /tmp/test.yaml

# Verify the environments
RUN conda env list

# Copy application files
COPY app /app

# Install Detectron2 inside the test environment
RUN conda run -n test pip install /app/preprocessing/detectron2

# Default command (change this based on your needs)
CMD ["conda", "run", "-n", "test", "python", "app.py"]
